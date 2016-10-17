import java.io.*;
import java.util.Random;
import java.lang.StringBuffer;
import java.util.ArrayList;
import java.util.Scanner;
import java.lang.Boolean;
import weka.core.FastVector;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.converters.ArffSaver;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.filters.unsupervised.attribute.Reorder;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;

class project2{ 

    
    static AttributeSelection currentSelection = new AttributeSelection(); //current selected attributes
    static StringToWordVector filter = new StringToWordVector();
    static Instances TrainingRelation;
    static Instances TestingRelation;
   
    static PrintWriter testResults;
    static PrintWriter trainResults;
    
    //Read in documents of a certain class to build relation
    public static void readFile(String fileName, String className, Instances relation){
	FileInputStream fin;
 	BufferedReader buffer;
	String documentText = null;
	try{
	    fin = new FileInputStream(fileName);
	    buffer = new BufferedReader(new InputStreamReader(fin));
	    while((documentText = buffer.readLine()) != null){
		if(documentText.trim().length() != 0){
		    Instance instance = new Instance(2);
		    instance.setDataset(relation);
		    instance.setClassValue(className);
		    instance.setValue(relation.attribute(0),documentText);
		    relation.add(instance);
		}
	    }
	}
	catch(Exception e){
	    System.out.println("DID NOT WORK!");
	    System.out.println(e);
	}
    }

    //Feature Construction
    public static Instances featureConstruction(Instances data, Reorder reorderFilter,String[] options, String saveFileName) throws Exception{
	Instances newInstances = Filter.useFilter(data, filter);
	reorderFilter.setInputFormat(newInstances);
	newInstances = Filter.useFilter(newInstances, reorderFilter);
	saveData(newInstances, saveFileName);
	return newInstances;
    }
    
    //Feature Selection
    static AttributeSelection featureSelection(Instances data, ASSearch search, ASEvaluation eval)throws Exception{
	AttributeSelection attrSelection = new AttributeSelection();
	attrSelection.setEvaluator(eval);
	attrSelection.setSearch(search);
	attrSelection.SelectAttributes(data);
	return attrSelection;
    }
    
    //Select k features using GainRatio
    static Instances GainRatio(Instances data, String saveFileName, int k) throws Exception{
	GainRatioAttributeEval evalGain = new GainRatioAttributeEval();
	Ranker searchGain = new Ranker();
	searchGain.setNumToSelect(k);
	currentSelection = featureSelection(data, searchGain, evalGain);
	Instances reducedData = removeSelection(data, currentSelection.selectedAttributes());
	saveData(reducedData, saveFileName);
	return reducedData;
    }
    
    //Select k feature using Chi Squared
    static Instances ChiSquared(Instances data, String saveFileName, int k) throws Exception{
	ChiSquaredAttributeEval evalChi = new ChiSquaredAttributeEval();
	Ranker searchChi = new Ranker();
	searchChi.setNumToSelect(k);
	currentSelection = featureSelection(data, searchChi, evalChi);
	Instances reducedData = removeSelection(data, currentSelection.selectedAttributes());
	saveData(reducedData, saveFileName);
	return reducedData;
    }
    
    //Remove selected attributes
    static Instances removeSelection(Instances data, int[] attributes)throws Exception{
	Remove remove = new Remove(); //set up the filter for removing attributes
	remove.setAttributeIndicesArray(attributes);
	remove.setInvertSelection(true);//retain the selected,remove all others
	remove.setInputFormat(data);
	return Filter.useFilter(data, remove);
    }

    static double[] buildClassifier(Classifier classifier, Instances trainData, Instances testData)throws Exception{
	classifier.buildClassifier(trainData);
	trainResults.println(classifier);
	
	Evaluation eval = new Evaluation(trainData);
	double[] predictions = eval.evaluateModel(classifier, testData);
	testResults.println(eval.toSummaryString());
	testResults.println(eval.toMatrixString());
	return predictions;
    }

    //saves instances to arff file
    public static void saveData(Instances saveInstances, String outputFile) throws Exception{
	ArffSaver saver = new ArffSaver();
	saver.setInstances(saveInstances);
	saver.setFile(new File(outputFile));
	saver.writeBatch();
    }

    // class Names
    static FastVector createClassTypes(ArrayList<String> classNames){
	FastVector classTypes = new FastVector();
	for(int i=0;i<classNames.size();++i){
	    if(!classTypes.contains(classNames.get(i)))
		classTypes.addElement(classNames.get(i));
	}
	return classTypes;
    }
    
    public static void main(String[] args){
	ArrayList<String> trainingFiles = new ArrayList<String>();
	ArrayList<String> testingFiles = new ArrayList<String>();
	ArrayList<String> classNames = new ArrayList<String>();
	Scanner input = new Scanner(System.in);

	int k = Integer.parseInt(args[0]); // number of features to select
	
	boolean done = false;
	do{
	    System.out.println("Enter in a training file, testing file, and a class type for a particular class");
	    System.out.print("Training File: ");
	    trainingFiles.add(input.nextLine());
	    System.out.print("Testing File: ");
	    testingFiles.add(input.nextLine());
	    System.out.print("Class Type: ");
	    classNames.add(input.nextLine());
	    System.out.println("Input more data?");
	    System.out.print("y/n: ");
	    if(input.nextLine().equals("n"))
		done = true;
	    
	}while(!done);
	
	//feature construction files

	//files contain features constructed using stemming and stop words 
	String fcStemStopTrain = "Class1NameFV-1.arff"; 
	String fcStemStopTest = "Class2NameFV-1.arff"; 
	//files contain feautres constructed using TF-IDF
	String fc_TFIDF_Train = "Class1NameFV-2.arff";
	String fc_TFIDF_Test = "Class2NameFV-2.arff"; 
	
	//feature selection files

	//files contain features selected using Chi Square
	String fsTrainingChi_SS = "Class1NameFV-1-k.arff";
	String fsTestingChi_SS = "Class2NameFV-1-k.arff";

	//files contain feautes selected using Gain Ratio
	String fsTrainingGain_TFIDF = "Class1NameFV-2-k.arff";
	String fsTestingGain_TFIDF = "Class2NameFV-2-k.arff";
	
      

	
	// creat class Names
	FastVector classTypes = createClassTypes(classNames);
	
	//String document attributes
	FastVector attributes = new FastVector();
	Attribute document = new Attribute("document", (FastVector)null);
	attributes.addElement(document);	
	attributes.addElement(new Attribute("className", classTypes));
	
	//Create relations 
	TrainingRelation = new Instances("TrainingData", attributes, 0);
	TestingRelation = new Instances("TestingData",attributes,0);
	TrainingRelation.setClassIndex(1);
	TestingRelation.setClassIndex(1);
	for(int i=0;i<trainingFiles.size();++i)
	    readFile(trainingFiles.get(i), classNames.get(i), TrainingRelation);
	for(int i=0;i<testingFiles.size();++i)
	    readFile(testingFiles.get(i), classNames.get(i), TestingRelation);


	try{
	    
	    //Reorders Instances to have the class set to the last attribute
	    Reorder reorderFilter = new Reorder();
	    String[] reorderOptions = {"-R"};
	    
	    //Feature construction by stemming with Lovins Stemmer and using stop words with Rainbow
	    String[] StemStop_options = 
		{"-stemmer", "weka.core.stemmers.LovinsStemmer", 
		 "-stopwords-handler","weka.core.stopwords.Rainbow"};
	    filter.setOptions(StemStop_options);
	    filter.setInputFormat(TrainingRelation);
	    Instances TrainingStemStop = featureConstruction(TrainingRelation, reorderFilter, StemStop_options, fcStemStopTrain);
	    Instances TestingStemStop =  featureConstruction(TestingRelation, reorderFilter, StemStop_options, fcStemStopTest);
	    
	    //Feature construction by TF-IDF with minumum frequency of 50
	    String[] TFIDF_options = {"-I","-M", "100", "-O"};
	    filter.setOptions(TFIDF_options);
	    filter.setInputFormat(TrainingRelation);
	    Instances TrainingTFIDF = featureConstruction(TrainingRelation, reorderFilter, TFIDF_options, fc_TFIDF_Train);
	    Instances TestingTFIDF =  featureConstruction(TestingRelation, reorderFilter, TFIDF_options, fc_TFIDF_Test);

	    Instances TrainingGain_TFIDF = GainRatio(TrainingTFIDF, fsTrainingGain_TFIDF, k);
	    int[] selattr = currentSelection.selectedAttributes();
	    Instances TestingGain_TFIDF = removeSelection(TestingTFIDF, selattr);
	    saveData(TestingGain_TFIDF, fsTestingGain_TFIDF);
	    	   
	    //Feature Selection by using Chi Squared
	    Instances TrainingChi_SS = ChiSquared(TrainingStemStop, fsTrainingChi_SS, k);
	    selattr  = currentSelection.selectedAttributes();
	    Instances TestingChi_SS = removeSelection(TestingStemStop, selattr);
	    saveData(TestingChi_SS, fsTestingChi_SS);
	    
	    
	    //Classification   
	    testResults = new PrintWriter("TestResults.txt");
	    trainResults = new PrintWriter("TrainResults.txt");
	    trainResults.println("==================== Feature Construction Method: Gain Ratio ====================");
	    trainResults.println("========== Feature Selection Method: TF-IDF ==========");
	    testResults.println("==================== Feature Construction Method: Gain Ratio ====================");
	    testResults.println("========== Feature Selection Method: TF-IDF ==========");
	    testResults.println("===== J48 Tree =====");
	    buildClassifier(new J48(), TrainingGain_TFIDF, TestingGain_TFIDF);
	    testResults.println("===== Naive Bayes =====");
	    buildClassifier(new NaiveBayes(), TrainingGain_TFIDF, TestingGain_TFIDF);
	    testResults.println("===== SMO =====");
	    buildClassifier(new SMO(), TrainingGain_TFIDF, TestingGain_TFIDF);
	    
	    trainResults.println("==================== Feature Construction Method: Stop Words and Stemming ====================");
	    trainResults.println("========== Feature Selection Method: Chi Square ==========");
	    testResults.println("==================== Feature Construction Method: Stop Words and Stemming ====================");
	    testResults.println("========== Feature Selection Method: Chi Square ==========");
	    testResults.println("===== J48 Tree =====");
	    buildClassifier(new J48(), TrainingChi_SS, TestingChi_SS);
	    testResults.println("===== Naive Bayes =====");
	    buildClassifier(new NaiveBayes(), TrainingChi_SS, TestingChi_SS);
	    testResults.println("===== SMO =====");
	    buildClassifier(new SMO(), TrainingChi_SS, TestingChi_SS);
	    
	    testResults.close();
	    trainResults.close();
	}
	catch(Exception e){
	    System.out.println("Filtering Failed");
	    System.out.println(e);
	    System.exit(1);
	}
    }
}
	
