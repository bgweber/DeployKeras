import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.handler.AbstractHandler;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
/**
 * Uses Jetty to deploy a Keras model. The service can be called as follows:
 * http://localhost:8080/?G1=1&G2=0&G3=1&G4=1&G5=0&G6=1&G7=0&G8=1&G9=1&G10=1
 */
public class JettyDL4J extends AbstractHandler {

	/** the model loaded from Keras **/
	private MultiLayerNetwork model;
	
	/** the number of input parameters in the Keras model **/ 
	private static int inputs = 10;		
	
	/** launch a web server on port 8080 */
	public static void main(String[] args) throws Exception {
		Server server = new Server(8080);
        server.setHandler(new JettyDL4J());
        server.start();
        server.join();
    }
	
	/** Loads the Keras Model **/
	public JettyDL4J() throws Exception {
		String simpleMlp = new ClassPathResource("games.h5").getFile().getPath();
		model = KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);
	}
		
	/** Returns a prediction for the passed in data set **/
	public void handle(String target,Request baseRequest, HttpServletRequest request, 
			HttpServletResponse response) throws IOException, ServletException {

		// create a dataset from the input parameters 
		INDArray features = Nd4j.zeros(inputs);
		for (int i=0; i<inputs; i++) {
			features.putScalar(new int[] {i}, Double.parseDouble(baseRequest.getParameter("G" + (i + 1))));
		}
        				
        // output the estimate
		double prediction = model.output(features).getDouble(0);	
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().println("Prediction: " + prediction);
        baseRequest.setHandled(true);
    }        	
}
