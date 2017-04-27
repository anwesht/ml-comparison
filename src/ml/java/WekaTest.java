// Java Program to automatically train on the given input data
// for various parameters of different classifiers like
// Random forest, decision tree, naive bayes, nearest neighbor, SVM
// We calculate the accuracy and AUC and write it to a tab separated
// file called output.tsv
package ml.java;

import java.util.Arrays;
import java.util.Random;
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

public class WekaTest {
    public static String FILE_PATH = "/Users/atuladhar/wekafiles/my_data/iris.arff";
    public static int NUM_REPLICATIONS = 5;
    public static int NUM_FOLDS = 2;
    public static int BASE_SEED = 2017;
    //************************************ Random forest ******************************//
    // private static evalRandomForest () 
    // {
    //     RandomForest modelRF = new RandomForest();
    //     Evaluation eval = new Evaluation(data);
    //     // Parameters for this classifier
    //     String[] options = new String[6];
    //     options[0] = "-P";  // Size of each bag, as a percentage of the training set size. (default 100)
    //     options[1] = "100";
    //     options[2] = "-I";  // Number of iterations. (current value 100)
    //     options[3] = "100";
    //     options[4] = "-K";  // Number of attributes to randomly investigate. (default 0) (<1 = int(log_2(#predictors)+1)).
    //     options[5] = "0";

    //     // Different combination of parameters
    //     int[] SizeOfBag = {90, 100};
    //     int[] NumberOfIteration = {100, 200};
    //     int[] NumberOfAttribute = {10, 15};
    //     for (int P : SizeOfBag) {
    //         for (int I : NumberOfIteration) {
    //             for (int K : NumberOfAttribute) {
    //                 options[0] = "-P";  // Size of each bag
    //                 options[1] = Integer.toString(P);
    //                 options[2] = "-I";  // Number of iterations
    //                 options[3] = Integer.toString(I);
    //                 options[4] = "-K";  // Number of attributes to randomly investigate.
    //                 options[5] = Integer.toString(K);

    //                 OutputFile.print("Random Forest\t"
    //                         + Arrays.toString(options));
    //                 // Set parameters for the classifier
    //                 modelRF.setOptions(options);
    //                 // Do FOLD fold cross validation
    //                 eval.crossValidateModel(modelRF, data, NUM_FOLDS, randomSeed);
    //                 // Calculate overall accuracy and AUC of current classifier
    //                 accuracy = eval.pctCorrect();
    //                 AUC = eval.weightedAreaUnderROC();
    //                 // Print current classifier's name and accuracy
    //                 OutputFile.println("\t" + String.format("%.2f%%", accuracy)
    //                         + "\t" + String.format("%.3f", AUC));
    //             }
    //         }
    //     }
    // }

    // //************************************ Nearest Neighbor ******************************//
    // private static evalNearestNeighbour () 
    // {
    //     IBk modelNN = new IBk();
    //     Evaluation eval = new Evaluation(data);
    //     // Parameters for this classifier
    //     String[] options = new String[2];
    //     options[0] = "-K";  // Number of nearest neighbours (k) used in classification. (Default = 1)
    //     options[1] = "1";

    //     // Different combination of parameters
    //     int[] NumberOfNN = {1, 2, 4, 8};
    //     for (int K : NumberOfNN) {
    //         options[0] = "-K";  // Number of nearest neighbours
    //         options[1] = Integer.toString(K);

    //         OutputFile.print("Nearest Neighbor\t"
    //                 + Arrays.toString(options));
    //         // Set parameters for the classifier
    //         modelNN.setOptions(options);
    //         // Do 10-split cross validation
    //         eval.crossValidateModel(modelNN, data, 10, new Random(1));
    //         // Calculate overall accuracy and AUC of current classifier
    //         accuracy = eval.pctCorrect();
    //         AUC = eval.weightedAreaUnderROC();
    //         // Print current classifier's name and accuracy
    //         OutputFile.println("\t" + String.format("%.2f%%", accuracy)
    //                 + "\t" + String.format("%.3f", AUC));
    //     }
    // }

    // //************************************ Decision tree ******************************//
    // private static evalDecisionTree () 
    // {
    //     J48 modelDT = new J48();
    //     Evaluation eval = new Evaluation(data);
    //     // Parameters for this classifier
    //     // String[] options = new String[6];

    //     OutputFile.print("Decision tree\tNone");
    //     // Set parameters for the classifier
    //     // modelDT.setOptions(options);
    //     // Do 10-split cross validation
    //     eval.crossValidateModel(modelDT, data, NUM_FOLDS, new Random(1));
    //     // Calculate overall accuracy and AUC of current classifier
    //     accuracy = eval.pctCorrect();
    //     AUC = eval.weightedAreaUnderROC();
    //     // Print current classifier's name and accuracy
    //     OutputFile.println("\t" + String.format("%.2f%%", accuracy)
    //             + "\t" + String.format("%.3f", AUC));
    // }

    // //************************************ Naive Bayes ******************************//
    // private static evalNaiveBayes () {
    //     NaiveBayes modelNB = new NaiveBayes();
    //     Evaluation eval = new Evaluation(data);
    //     // Parameters for this classifier
    //     // String[] options = new String[6];

    //     OutputFile.print("Naive Bayes\tNone");
    //     // Set parameters for the classifier
    //     // modelDT.setOptions(options);
    //     // Do 10-split cross validation
    //     eval.crossValidateModel(modelNB, data, 10, new Random(1));
    //     // Calculate overall accuracy and AUC of current classifier
    //     accuracy = eval.pctCorrect();
    //     AUC = eval.weightedAreaUnderROC();
    //     // Print current classifier's name and accuracy
    //     OutputFile.println("\t" + String.format("%.2f%%", accuracy)
    //             + "\t" + String.format("%.3f", AUC));
    // }

    // //************************************ Support vector machine ******************************//
    // private static evalSVM() {
    //     SMO modelSVM = new SMO();
    //     Evaluation eval = new Evaluation(data);
    //     // Parameters for this classifier
    //     String[] options = new String[2];
    //     options[0] = "-C";  // The complexity constant C. (default 1)
    //     options[1] = "1.0";

    //     // Different combination of parameters
    //     double[] ComplexityConstant = {0.01, 0.1, 1.0, 10.0};
    //     for (double C : ComplexityConstant) {
    //         options[0] = "-C";  // The complexity constant
    //         options[1] = Double.toString(C);

    //         OutputFile.print("SVM\t"
    //                 + Arrays.toString(options));
    //         // Set parameters for the classifier
    //         modelSVM.setOptions(options);
    //         // Do 10-split cross validation
    //         eval.crossValidateModel(modelSVM, data, 10, new Random(1));
    //         // Calculate overall accuracy and AUC of current classifier
    //         accuracy = eval.pctCorrect();
    //         AUC = eval.weightedAreaUnderROC();
    //         // Print current classifier's name and accuracy
    //         OutputFile.println("\t" + String.format("%.2f%%", accuracy)
    //                 + "\t" + String.format("%.3f", AUC));
    //     }
    // }

    /*private static Evaluation evaluateClassifier (Classifier c, Instances randData, int n) {
        Evaluation eval = new Evaluation(randData);

        Instances train = randData.trainCV(NUM_FOLDS, n);
        Instances test = randData.testCV(NUM_FOLDS, n);
      
        // build and evaluate classifier
        c.buildClassifier(train);
        eval.evaluateModel(c, test);
        return eval;
    }*/


    public static void main(String[] args) throws Exception {
    	

    	if(args.length == 1) {
    		FILE_PATH = args[0];
    		System.out.println("File is: " + FILE_PATH);    		
    	}

        // Variables to store the accuracy and area under ROC
        double accuracy;
        double AUC;

        // Create output file and write header to it
        PrintWriter OutputFile = new PrintWriter("output.tsv", "UTF-8");
        OutputFile.println("File name: " + FILE_PATH);
        OutputFile.println("Classifier\tParameter\tAccuracy\tAUC");

        // Load the data
        DataSource source = new DataSource(FILE_PATH);
        Instances data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1){
            System.out.println("here");
            data.setClassIndex(data.numAttributes() - 1);
        }

        J48 modelDT = new J48();
        
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
                // Evaluation eval = evaluateClassifier(modelDT, randData, n);
                Evaluation eval = new Evaluation(randData);

        Instances train = randData.trainCV(NUM_FOLDS, n);
        Instances test = randData.testCV(NUM_FOLDS, n);
      
        // build and evaluate classifier
        modelDT.buildClassifier(train);
        eval.evaluateModel(modelDT, test);
                
                // Calculate overall accuracy and AUC of current classifier
                OutputFile.print("Decision tree\tNone");
                accuracy = eval.pctCorrect();
                AUC = eval.weightedAreaUnderROC();
                // Print current classifier's name and accuracy
                OutputFile.println("\t" + String.format("%.2f%%", accuracy)
                        + "\t" + String.format("%.3f", AUC));

                
                //call random forest
                //call nearest neighbour
                //call decision tree
                //call Naive Bayes
                //call SVM
            }        
        }
        
        OutputFile.close();
    }

 /*   public static void main(String[] args) throws Exception {
        // loads data and set class index
        Instances data = DataSource.read(Utils.getOption("t", args));
        String clsIndex = Utils.getOption("c", args);
        if (clsIndex.length() == 0)
          clsIndex = "last";
        if (clsIndex.equals("first"))
          data.setClassIndex(0);
        else if (clsIndex.equals("last"))
          data.setClassIndex(data.numAttributes() - 1);
        else
          data.setClassIndex(Integer.parseInt(clsIndex) - 1);

        // classifier
        String[] tmpOptions;
        String classname;
        tmpOptions     = Utils.splitOptions(Utils.getOption("W", args));
        classname      = tmpOptions[0];
        tmpOptions[0]  = "";
        Classifier cls = (Classifier) Utils.forName(Classifier.class, classname, tmpOptions);

        // other options
        int runs  = Integer.parseInt(Utils.getOption("r", args));
        int folds = Integer.parseInt(Utils.getOption("x", args));

        // perform cross-validation
        for (int i = 0; i < runs; i++) {
          // randomize data
          int seed = i + 1;
          Random rand = new Random(seed);
          Instances randData = new Instances(data);
          randData.randomize(rand);
          if (randData.classAttribute().isNominal())
            randData.stratify(folds);

          Evaluation eval = new Evaluation(randData);
          for (int n = 0; n < folds; n++) {
            Instances train = randData.trainCV(folds, n);
            Instances test = randData.testCV(folds, n);
            // the above code is used by the StratifiedRemoveFolds filter, the
            // code below by the Explorer/Experimenter:
            // Instances train = randData.trainCV(folds, n, rand);

            // build and evaluate classifier
            Classifier clsCopy = Classifier.makeCopy(cls);
            clsCopy.buildClassifier(train);
            eval.evaluateModel(clsCopy, test);
          }

          // output evaluation
          System.out.println();
          System.out.println("=== Setup run " + (i+1) + " ===");
          System.out.println("Classifier: " + cls.getClass().getName() + " " + Utils.joinOptions(cls.getOptions()));
          System.out.println("Dataset: " + data.relationName());
          System.out.println("Folds: " + folds);
          System.out.println("Seed: " + seed);
          System.out.println();
          System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false));
        }
  }*/
}