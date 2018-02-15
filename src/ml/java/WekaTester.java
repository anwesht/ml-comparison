// Java Program to automatically train on the given input data
// for various parameters of different classifiers like
// Random forest, decision tree, naive bayes, nearest neighbor, SVM
// We calculate the accuracy and AUC and write it to a tab separated
// file called output.tsv
// Compile from src directory: javac -d ../bin ml/java/WekaTester.java
// Run from bin directory: java ml.java.WekaTester
package ml.java;

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

import ml.java.WekaCrossValidation;
import ml.java.Stats;

public class WekaTester {
    public static String FILE_PATH = "/Users/atuladhar/wekafiles/my_data/iris.arff";
   
    public static void main(String[] args) throws Exception {
    	if(args.length == 1) {
    		FILE_PATH = args[0];
    		System.out.println("File is: " + FILE_PATH);    		
    	}
    
        // Load the data
        DataSource source = new DataSource(FILE_PATH);
        Instances data = source.getDataSet();

        String outputFile = FILE_PATH.substring(FILE_PATH.lastIndexOf('/') + 1);
        List<Classifier> classifierList = new ArrayList<Classifier>();
        WekaCrossValidation cv = new WekaCrossValidation(5, 2, outputFile);

        //********************************** Decision Tree ************************************//
        J48 modelDT = new J48();
        // classifierList.add(modelDT);

        //********************************** Random Forest ************************************//
        RandomForest modelRF = new RandomForest();
        // Parameters for this classifier
        String[] rfOptions = new String[6];
        rfOptions[0] = "-P";  // Size of each bag, as a percentage of the training set size. (default 100)
        rfOptions[1] = "100";
        rfOptions[2] = "-I";  // Number of iterations. (current value 100)
        rfOptions[3] = "200";
        rfOptions[4] = "-K";  // Number of attributes to randomly investigate. (default 0) (<1 = int(log_2(#predictors)+1)).
        rfOptions[5] = "15";
        modelRF.setOptions(rfOptions);
        // classifierList.add(modelRF);

        //********************************** SVM ************************************//
        SMO modelSVM = new SMO();       //SMO = Sequential Minimal Optimization algorithm for training a support vector classifier.
        // Parameters for this classifier
        String[] svmOptions = new String[2];
        svmOptions[0] = "-C";  // The complexity constant C. (default 1)
        svmOptions[1] = "1.0";
        modelSVM.setOptions(svmOptions);
        classifierList.add(modelSVM);

        // For given data, Cross Validate all the classifiers.
        cv.crossValidate(classifierList, data);
    }
}