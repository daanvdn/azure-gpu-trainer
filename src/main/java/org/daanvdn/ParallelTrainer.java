package org.daanvdn;

import com.google.common.base.Charsets;
import com.google.common.base.Stopwatch;
import com.google.common.base.Throwables;
import com.google.common.io.Files;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.deeplearning4j.datasets.iterator.file.FileDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.kohsuke.args4j.*;
import org.kohsuke.args4j.spi.EnumOptionHandler;
import org.kohsuke.args4j.spi.ExplicitBooleanOptionHandler;
import org.kohsuke.args4j.spi.FileOptionHandler;
import org.kohsuke.args4j.spi.Setter;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.cache.InMemoryDataSetCache;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Slf4j
public class ParallelTrainer {

    private static final long AVALIABLE_DEVICE_MEMORY = 17179869184L;
    @Option(name = "-epochs", required = true)
    private Integer epochs;
    @Option(name = "-train", required = true, handler = FileOptionHandler.class)
    private File trainDir;
    @Option(name = "-test", required = true, handler = FileOptionHandler.class)
    private File testDir;
    @Option(name = "-labels", required = true, handler = FileOptionHandler.class)
    private File labelsFile;
    @Option(name = "-config", required = true)
    private File neuralNetConf;
    @Option(name = "-workers", required = true)
    private Integer workers;
    @Option(name = "-avg", required = true)
    private Integer averagingFrequency;
    @Option(name = "-prefetch", required = false)
    private Integer prefetchBuffer = 16 * Nd4j.getAffinityManager().getNumberOfDevices();
    @Option(name = "-report", required = false, handler = ExplicitBooleanOptionHandler.class)
    private Boolean reportScore = true;
    @Option(name = "-gcWindow", required = false)
    private int gcWindow = 100000;
    @Option(name = "-autoGc", required = false, handler = ExplicitBooleanOptionHandler.class)
    private Boolean autoGc = false;
    @Option(name = "-tmpDir", required = true, handler = FileOptionHandler.class)
    private File tmpDir;
//    @Option(name = "-parties" ,required = true )
//    private Integer parties;

    @Option(name = "-cacheDir", required = true, handler = FileOptionHandler.class)
    private File cacheDir;
    //    @Option(name = "-mode", required = false, handler = TrainingModeOptionHandler.class)
    private ParallelWrapper.TrainingMode trainingMode = ParallelWrapper.TrainingMode.AVERAGING;


    private DataSetIterator trainData;
    private DataSetIterator testData;
    private MultiLayerConfiguration multiLayerConfiguration;
    private List<String> labels;
    private MultiLayerNetwork net;
    private final PortListener portListener = new PortListener();
    private final Evaluator evaluator;
    private static final String DURATION_FORMAT = "HHHH 'hrs' mm 'mins' ss 'sec'";


    public ParallelTrainer(String[] args) throws IOException {
        parseCmdLineArgs(this, args);
        portListener.start();
        evaluator = new Evaluator(tmpDir, epochs);
        Nd4j.getMemoryManager().togglePeriodicGc(autoGc);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        labels = Files.readLines(labelsFile, Charsets.UTF_8);
        trainData = createCachingDataSetIterator(trainDir);
        testData = createCachingDataSetIterator(testDir);
        multiLayerConfiguration = MultiLayerConfiguration.fromJson(Files.toString(neuralNetConf, Charsets.UTF_8));
        net = new MultiLayerNetwork(multiLayerConfiguration);
        net.init();
        net.setListeners(new ScoreIterationListener());
    }

    private DataSetIterator createFileDataSetIterator(File dir) {
        FileDataSetIterator baseDataSetiterator = new FileDataSetIterator(dir);
        baseDataSetiterator.setLabels(labels);
        return baseDataSetiterator;
    }


    private DataSetIterator createCachingDataSetIterator(File dir) {
        FileDataSetIterator dataSetIterator = new FileDataSetIterator(dir);
        return new CachingDataSetIterator(dataSetIterator, new InMemoryDataSetCache());
    }


    public void run() throws IOException {

      /*  WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.OVER_TIME) // <-- this option makes workspace learning after first loop
                .build();*/
        ParallelWrapper parallelWrapper = getParallelWrapper();

        for (int epoch = 0; epoch < epochs; epoch++) {

            Stopwatch trainTimer = Stopwatch.createStarted();
            log.info("Started training epoch {} of {}", epoch, epochs);
            parallelWrapper.fit(trainData);
            String trainElapsed = DurationFormatUtils.formatDuration(trainTimer.elapsed(TimeUnit.MILLISECONDS), DURATION_FORMAT, false);
            trainData.reset();
            log.info("Finished training epoch {} of {} in {}", epoch, epochs, trainElapsed);
            evaluator.evaluate(net, epoch, labels, testData);

        }
        portListener.stop();

    }


    private ParallelWrapper getParallelWrapper() {


        return new ParallelWrapper.Builder<>(net).workers(workers)
                .averagingFrequency(averagingFrequency)
                .prefetchBuffer(prefetchBuffer)
                .reportScoreAfterAveraging(reportScore)
                .averageUpdaters(true)
                .trainingMode(trainingMode)
//                .gradientsAccumulator(new EncodedGradientsAccumulator(parties, 1e-3))
                .build();
    }

    private class TrainingModeOptionHandler extends EnumOptionHandler<ParallelWrapper.TrainingMode> {

        public TrainingModeOptionHandler(CmdLineParser parser, OptionDef option, Setter<? super ParallelWrapper.TrainingMode> setter) {
            super(parser, option, setter, ParallelWrapper.TrainingMode.class);
        }
    }

    private void parseCmdLineArgs(Object bean, String[] args) {
        CmdLineParser clParser = null;
        try {
            ParserProperties properties = ParserProperties.defaults();
            properties.withUsageWidth(100);
            clParser = new CmdLineParser(bean, properties);
            clParser.parseArgument(args);

            List<Field> optionFields = FieldUtils.getFieldsListWithAnnotation(bean.getClass(),
                    Option.class);
            List<String> options = new ArrayList<>();


            for (Field optionField : optionFields) {
                optionField.setAccessible(true);
                try {
                    Option annotation = optionField.getAnnotation(Option.class);
                    String annotationName = annotation.name();
                    Object value = optionField.get(bean);
                    String valueStr = value == null ? "null" : value.toString();
                    options.add(annotationName);
                    options.add(valueStr);

                } catch (IllegalAccessException e) {
                    throw new RuntimeException(e);
                }
            }

            log.info("Started {} with the following program options: {}", bean.getClass().getSimpleName(),
                    options.stream().collect(Collectors.joining(" ")));


        } catch (CmdLineException e) {
            log.error(e.getMessage());
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            PrintStream ps = new PrintStream(baos);
            clParser.printUsage(ps);
            String msg = new String(baos.toByteArray(), Charsets.UTF_8);
            log.error(msg);
            Throwables.propagate(e);
        } catch (IllegalStateException e) {
            log.error(Throwables.getStackTraceAsString(e));
            Throwables.propagate(e);
        }
    }


    private class PortListener {

        private Process p;

        private void start() throws IOException {
            p = Runtime.getRuntime().exec("nc -l -p 8081 localhost");
        }


        private void stop() {
            p.destroy();
        }
    }


    public static void main(String[] args) throws IOException {


//        Nd4j.setDataType(DataBuffer.Type.HALF);
        CudaEnvironment.getInstance().getConfiguration()
                // key option enabled
                .allowMultiGPU(true)

                // we're allowing larger memory caches
//                .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)

                // cross-device access is used for faster model averaging over pcie
                .allowCrossDeviceAccess(true);


        new ParallelTrainer(args).run();

    }

}
