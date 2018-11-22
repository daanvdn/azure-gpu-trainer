package org.daanvdn;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.stream.Collectors;

@Slf4j
public class Evaluator {

    private static final String DURATION_FORMAT = "HHHH 'hrs' mm 'mins' ss 'sec'";

    private final File directory;
    private final int epochs;
    @Getter
    private double bestAvgF1 = 0.0;
    @Getter
    private int bestEpoch = -1;
    @Getter
    private File bestModel = null;
    @Getter
    private EvaluationBinary bestEvaluation;

    public Evaluator(File directory, int epochs) {

        this.directory = directory;
        this.epochs = epochs;
        if (!directory.exists()) {
            log.info("Target Directory '{}' does not exist. Creating it now!", directory.getAbsolutePath());
            directory.mkdirs();
        } else {
            File[] files = directory.listFiles(File::isFile);
            if (files != null && files.length > 0) {
                log.info("Target directory '{}' contains {} files. These will be deleted!",
                        directory.getAbsolutePath(),
                        files.length);
                String filePaths = Arrays.stream(files).map(File::getAbsolutePath).collect(Collectors.joining(";"));
                Lists.newArrayList(files).forEach(File::delete);
                log.info("Finished deleting the following files: {}", filePaths);
            }
        }
    }


    public void evaluate(MultiLayerNetwork net, int epoch, List<String> labels, DataSetIterator testData) throws IOException {


        try {
            log.info("Started testing epoch {} of {}", epoch, epochs);

            Stopwatch testTimer = Stopwatch.createStarted();
            EvaluationBinary evaluationBinary = net.doEvaluation(testData, new EvaluationBinary())[0];
            evaluationBinary.setLabels(labels);
            String stats = evaluationBinary.stats();
            log.info("\n{}", stats);
            testData.reset();
            String testElapsed = DurationFormatUtils.formatDuration(testTimer.elapsed(TimeUnit.MILLISECONDS), DURATION_FORMAT, false);
            log.info("Finished testing epoch {} of {} in {}", epoch, epochs, testElapsed);
            double avgF1 = evaluationBinary.averageF1();
            if (!(Double.isNaN(avgF1) && avgF1 > bestAvgF1) || epoch == 0) {
                bestAvgF1 = avgF1;
                bestEpoch = epoch;
                bestEvaluation = evaluationBinary;
                saveModel(net, epoch);
            }
            Supplier<String> getBestEpochInfo = () -> {
                if (bestEpoch == epoch) {
                    return "Best epoch so far!";
                } else {
                    double delta = bestAvgF1 - avgF1;
                    return String.format("%f worse than best epoch (%d, f1=%f)", delta, bestEpoch, bestAvgF1);
                }
            };

            log.info("Evaluation epoch {}\n{}\nAverage F1={}. {}", epoch, stats, avgF1, getBestEpochInfo.get());

        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }


    private void saveModel(MultiLayerNetwork multiLayerNetwork, int epoch) throws IOException, InterruptedException {
        String formattedDate = getCurrentDate("_");
        File outputFile = new File(directory, String.format("MultiLayerNetwork_epoch%d_%s.bin", epoch, formattedDate));
        ModelSerializer.writeModel(multiLayerNetwork, outputFile, true);
        log.info("Written model to file '{}'", outputFile.getAbsolutePath());
        if (bestModel != null) {
            Files.deleteIfExists(bestModel.toPath());
            Thread.sleep(1000);
        }
        bestModel = outputFile;

    }

    private static String getCurrentDate(String sep) {
        String pattern = String.format("yyyy%sMM%sdd%sHH%smm%sss", sep, sep, sep, sep, sep);
        DateFormat dateFormat = new SimpleDateFormat(pattern);
        Date date = new Date();
        return dateFormat.format(date);
    }


}
