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
    public static int BASE_SEED = 2017;

    public WekaCrossValidation(){
        this.NUM_REPLICATIONS = 5;
        this.NUM_FOLDS = 2;
    }

    public WekaCrossValidation(int num_replications, int num_folds) {
        this.NUM_REPLICATIONS = num_replications;
        this.NUM_FOLDS = num_folds;
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
    	// Variables to store the accuracy and area under ROC
        double accuracy;
        double AUC;

        // Create output file and write header to it
        PrintWriter OutputFile = new PrintWriter("output.tsv", "UTF-8");
        // OutputFile.println("File name: " + FILE_PATH);
        OutputFile.println("Classifier\tParameter\tAccuracy\tAUC");

        // Load the data
        /*DataSource source = new DataSource(FILE_PATH);
        Instances data = source.getDataSet();*/
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1){
            System.out.println("here");
            data.setClassIndex(data.numAttributes() - 1);
        }

        // J48 modelDT = new J48();
        
        //Do NUM_REPLICATIONS x NUM_FOLDS cross validation
        for (int i = 0; i < NUM_REPLICATIONS; i++){
            Random rand = new Random(BASE_SEED + i);   // create seeded number generator
            Instances randData = new Instances(data);   // create copy of original data
            randData.randomize(rand);         // randomize data with number generator
            if (randData.classAttribute().isNominal()){
                randData.stratify(NUM_FOLDS);       // Stratify nominal data
            }

            for(int n = 0; n < NUM_FOLDS; n++) {
                // Classifier clsCopy = Classifier.makeCopy((Classifier)modelDT);
                
                for(Classifier c : classifierList){
                    Evaluation eval = evaluateClassifier(c, randData, n);
                    
                    // Calculate overall accuracy and AUC of current classifier
                    OutputFile.print(c.getClass().getName() + "\tNone");
                    accuracy = eval.pctCorrect();
                    AUC = eval.weightedAreaUnderROC();
                    // Print current classifier's name and accuracy
                    OutputFile.println("\t" + String.format("%.2f%%", accuracy)
                            + "\t" + String.format("%.3f", AUC));
                }
            }        
        }
        
        OutputFile.close();
    }
}