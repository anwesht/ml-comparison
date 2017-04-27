// Java Program to automatically train on the given input data
// for various parameters of different classifiers like
// Random forest, decision tree, naive bayes, nearest neighbor, SVM
// We calculate the accuracy and AUC and write it to a tab separated
// file called output.tsv
package ml.java;

// import java.util.Arrays;
// import java.util.Random;
import java.util.*;
import java.io.PrintWriter;

import org.apache.commons.math3.stat.inference.TTest; 

import weka.classifiers.Classifier;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaCrossValidation {
    public int NUM_REPLICATIONS;
    public int NUM_FOLDS;
    public String OUTPUT_FILE;
    public static int BASE_SEED = 2017;

    public WekaCrossValidation(){
        this.NUM_REPLICATIONS = 5;
        this.NUM_FOLDS = 2;
    }

    public WekaCrossValidation(int num_replications, int num_folds) {
        this.NUM_REPLICATIONS = num_replications;
        this.NUM_FOLDS = num_folds;
    }

    public WekaCrossValidation(int num_replications, int num_folds, String outputFile) {
        this.NUM_REPLICATIONS = num_replications;
        this.NUM_FOLDS = num_folds;
        this.OUTPUT_FILE = outputFile;
    }

    private Evaluation evaluateClassifier (Classifier c, Instances randData, int n) throws Exception {
        Evaluation eval = new Evaluation(randData);

        Instances train = randData.trainCV(NUM_FOLDS, n);
        Instances test = randData.testCV(NUM_FOLDS, n);
      
        // build and evaluate classifier
        c.buildClassifier(train);
        eval.evaluateModel(c, test);
        return eval;
    }
    
    public void crossValidate(List<Classifier> classifierList, Instances data) throws Exception {
    // public Map<String, double[]> crossValidate(List<Classifier> classifierList, Instances data) throws Exception {
    	// Variables to store the accuracy, area under ROC, errorRate and t-statistics
        double accuracy;
        double AUC;
        double errorRate;
        TTest ttester = new TTest();

        if (data.classIndex() == -1){
            data.setClassIndex(data.numAttributes() - 1);
        }

        Map<String, double[]> errorsMap = new HashMap<String, double[]>();
        for (int i = 0; i < classifierList.size(); i++) {
            errorsMap.put(classifierList.get(i).getClass().getSimpleName(), new double[NUM_REPLICATIONS*NUM_FOLDS]);
        }


        // Create output file and write header to it
        PrintWriter OutputFile = new PrintWriter(OUTPUT_FILE + "_output.tsv", "UTF-8");        
        OutputFile.println("Performing " + Integer.toString(NUM_REPLICATIONS)
                    + " X " + Integer.toString(NUM_FOLDS) + " Cross Validation.");
        // OutputFile.println("Classifier\tParameter\tAccuracy\tAUC\terrorRate");

        //Do NUM_REPLICATIONS x NUM_FOLDS cross validation
        for (int i = 0; i < NUM_REPLICATIONS; i++){
            Random rand = new Random(BASE_SEED + i);   // create seeded number generator
            Instances randData = new Instances(data);   // create copy of original data
            randData.randomize(rand);         // randomize data with number generator
            if (randData.classAttribute().isNominal()){
                randData.stratify(NUM_FOLDS);       // Stratify nominal data
            }

            for(int n = 0; n < NUM_FOLDS; n++) {
                OutputFile.println("------------------------------------------------------");
                OutputFile.println("NUM_REPLICATION = " + Integer.toString(i+1)
                    + "  NUM_FOLD = " + Integer.toString(n+1));
                OutputFile.println("------------------------------------------------------");

                for(Classifier c : classifierList){
                    Evaluation eval = evaluateClassifier(c, randData, n);
                    
                    // Calculate overall accuracy and AUC of current classifier
                    OutputFile.print(c.getClass().getSimpleName() + "\n\t");
                    accuracy = eval.pctCorrect();
                    AUC = eval.weightedAreaUnderROC();
                    errorRate = eval.errorRate();

                    //Add errorRates to map.
                    errorsMap.get(c.getClass().getSimpleName())[i*n] = errorRate;

                    // System.out.println(c.getClass().getSimpleName()+ "  = " +eval.errorRate());
                    // System.out.println(errorsMap.get(c.getClass().getSimpleName())[i*n]);
                    // Print current classifier's name and accuracy
                    OutputFile.println("Accuracy = " + String.format("%.2f%%", accuracy)
                            + ",\t AUC = " + String.format("%.3f", AUC)+ ",\t Error Rate = " + String.format("%.3f", errorRate));
                }
            }        
        }

        OutputFile.println("\n\n\n\n");
        
        errorsMap.forEach((c1, errors1)-> {
            errorsMap.forEach((c2, errors2) -> {
                    if(!c1.equals(c2)) {
                        // System.out.println("t("+c1+", "+c2 + ") = "+ Double.toString(tstatistic));
                        OutputFile.println("------------------------------------------------------");
                        OutputFile.println("("+c1 + " Vs " + c2 + ")");
                        /* Computes a paired, 2-sample t-statistic based on the data in the input arrays. */
                        OutputFile.println("\tpairedT/t-statistic = " + Double.toString(ttester.pairedT(errors1, errors2)));
                        /* Returns the observed significance level, or p-value, associated with a paired, two-sample, two-tailed t-test based on the data in the input arrays. */
                        OutputFile.println("\tpairedTTest/observed significance level/p-value = " + Double.toString(ttester.pairedTTest(errors1, errors2)));
                        /** Performs a paired t-test evaluating the null hypothesis that the 
                            mean of the paired differences between sample1 and sample2 is 0 in favor of 
                            the two-sided alternative that the mean paired difference is not equal to 0, with significance level alpha.
                          */
                        OutputFile.println("\tpairedTTest null hypothesis (with alpha = 0.05) = "+ Boolean.toString(ttester.pairedTTest(errors1, errors2, 0.05)));
                        OutputFile.println("------------------------------------------------------");
                    }
                });
            });

        OutputFile.close();
        // return errorsMap;
    }
}