package org.daanvdn;

import com.google.common.base.Charsets;
import com.google.common.base.Throwables;
import com.google.common.io.Files;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.file.FileDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.kohsuke.args4j.*;
import org.kohsuke.args4j.spi.EnumOptionHandler;
import org.kohsuke.args4j.spi.FileOptionHandler;
import org.kohsuke.args4j.spi.Setter;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
public class ParallelTrainer {

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
    @Option(name = "-prefetch", required = true)
    private Integer prefetchBuffer;
    @Option(name = "-report", required = false)
    private Boolean reportScore = true;
//    @Option(name = "-mode", required = false, handler = TrainingModeOptionHandler.class)
    private ParallelWrapper.TrainingMode trainingMode = ParallelWrapper.TrainingMode.AVERAGING;


    private DataSetIterator trainData;
    private DataSetIterator testData;
    private MultiLayerConfiguration multiLayerConfiguration;
    private List<String> labels;
    private MultiLayerNetwork net;


    public ParallelTrainer(String[] args) throws IOException {
        parseCmdLineArgs(this, args);
        labels = Files.readLines(labelsFile, Charsets.UTF_8);
        trainData = createDataSetIterator(trainDir);
        testData = createDataSetIterator(testDir);
        multiLayerConfiguration = MultiLayerConfiguration.fromJson(Files.toString(neuralNetConf, Charsets.UTF_8));
        net = new MultiLayerNetwork(multiLayerConfiguration);
        net.init();
        net.setListeners(new ScoreIterationListener());
    }

    private DataSetIterator createDataSetIterator(File dir) {
        FileDataSetIterator baseDataSetiterator = new FileDataSetIterator(dir);
        baseDataSetiterator.setLabels(labels);
        return new AsyncDataSetIterator(baseDataSetiterator);
    }


    public void run() {

        WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.OVER_TIME) // <-- this option makes workspace learning after first loop
                .build();
        ParallelWrapper parallelWrapper = getParallelWrapper();

        for (int epoch = 0; epoch < epochs; epoch++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig,
                    "mem_ws_ID")) {
                log.info("Started training epoch {} of {}", epoch, epochs);
                parallelWrapper.fit(trainData);
                trainData.reset();
                log.info("Finished training epoch {} of {}", epoch, epochs);
                log.info("Started testing epoch {} of {}", epoch, epochs);
                EvaluationBinary evaluationBinary = net.doEvaluation(testData, new EvaluationBinary())[0];
                String stats = evaluationBinary.stats();
                log.info("\n{}", stats);
                testData.reset();
                log.info("Finished testing epoch {} of {}", epoch, epochs);


            }
        }

    }


    private ParallelWrapper getParallelWrapper() {


        return new ParallelWrapper.Builder<>(net).workers(workers)
                .averagingFrequency(averagingFrequency)
                .prefetchBuffer(prefetchBuffer)
                .reportScoreAfterAveraging(reportScore)
                .averageUpdaters(true)
                .trainingMode(trainingMode)
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

    public static void main(String[] args) throws IOException {


        new ParallelTrainer(args).run();

    }
}
