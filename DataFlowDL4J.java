import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.PTransform;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.values.PCollection;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;

public class DataFlowDL4J {
	
	/** The GCS project name */
	private static final String PROJECT_ID = "your_project_ID";

	/** The dataset name for the BigQuery output table */
	private static final String dataset = "dataflow";

	/** The table name for the BigQuery output table */
	private static final String table = "game_predictions"	;
	
	/** Provide an interface for setting theGCS temp location */
	interface Options extends PipelineOptions, Serializable {
		String getTempLocation();
	    void setTempLocation(String value);
	}

	public static void main(String[] args) throws Exception {

		// read the file with the records to score
		URL url = new URL("https://raw.githubusercontent.com/bgweber/DeployKeras/master/games-expand.csv");
		BufferedReader reader = new BufferedReader(new InputStreamReader(url.openStream()));
		reader.readLine();
		String line = reader.readLine();

		// create a TableRow for each CSV line 
		ArrayList<TableRow> records = new ArrayList<>();
		while (line != null) {
			String[] attributes = line.split(",");
			TableRow row = new TableRow();
			
			for (int i=0; i<10; i++) {
				row.set("G" + (i+1), attributes[i]);
			}
			
			row.set("actual", attributes[10]);
			records.add(row);
			line = reader.readLine();
		}
		
	    // create the schema for the results table
	    List<TableFieldSchema> fields = new ArrayList<>();
	    fields.add(new TableFieldSchema().setName("actual").setType("INT64"));
	    fields.add(new TableFieldSchema().setName("predicted").setType("FLOAT64"));
	    TableSchema schema = new TableSchema().setFields(fields);

			    
		// set up the dataflow pipeline 
	    DataFlowDL4J.Options options = PipelineOptionsFactory.fromArgs(args).withValidation().as(DataFlowDL4J.Options.class);
	    Pipeline pipeline = Pipeline.create(options);

	    // create a PCollection using the records from the CSV file 
	    pipeline.apply(Create.of(records))
	    
	    // apply the Keras model 
	    .apply("Keras Predict", new PTransform<PCollection<TableRow>, PCollection<TableRow>>() {

	    	// define a transform that loads the PMML specification and applies it to all of the records
	        public PCollection<TableRow> expand(PCollection<TableRow> input) {	    
	        	
	        	// load the model
	        	final MultiLayerNetwork model;
	    		final int inputs = 10;	
	        	try {
		    		String simpleMlp = new ClassPathResource("games.h5").getFile().getPath();
		    		model = KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);
	        	}
	            catch (Exception e) {
	            	throw new RuntimeException(e);
	            }
	        	
	            // create a DoFn for applying the PMML model to instances
	            return input.apply("To Predictions", ParDo.of(new DoFn<TableRow, TableRow>() {
	            	
	                @ProcessElement
	                public void processElement(ProcessContext c) throws Exception {
	                	TableRow row = c.element();
	                  
	                	// create the feature vector 
	                	INDArray features = Nd4j.zeros(inputs);
	                	for (int i=0; i<inputs; i++) {
	                		features.putScalar(new int[] {i}, Double.parseDouble(row.get("G" + (i+1)).toString()));
	                	}
	        		
	                	// get the prediction
	                	double estimate = model.output(features).getDouble(0);	

	                	// record the result
	                	TableRow prediction = new TableRow();
	                	prediction.set("actual", row.get("actual"));
	                	prediction.set("predicted", estimate);
	                	
	                	// output the prediction to the data flow pipeline
	                	c.output(prediction);
	                }
	            }));
	        }
	    })
	    // write the results to BigQuery 	
	    .apply(BigQueryIO.writeTableRows() .to(String.format("%s:%s.%s", PROJECT_ID, dataset, table))
	            .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED)
	            .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_TRUNCATE)
	            .withSchema(schema)
	    );
	    
	    // run the pipeline
	    pipeline.run();
	}
}
