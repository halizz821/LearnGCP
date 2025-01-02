#This code is a Flask application designed to expose two REST API endpoints: /ping and /predict. It allows users to test the service or send data for predictions

import flask #A lightweight web framework for Python used to create REST APIs and serve web applications.

app = flask.Flask(__name__) #  app is a web application object (application instance) created in your code (it is used in line 216-220 of original code). It is defined using a web framework such as Flask
                            #  Creates an instance of the Flask application.


############################################################################################
# The ping Endnote (for testing the API,whether it is working)
@app.route("/ping", methods=["POST"])  #Defines the route /ping. Accepts only POST requests. A POST request is a type of HTTP request used to send data from a client (e.g., your browser or a script) to a server. 
        
        # It's commonly used in REST APIs when you need to submit or upload data for processing.
def run_root() -> str:
    args = flask.request.get_json() or {} #Receives a JSON payload from the client. Example: POST /ping
                                                                                     #       Content-Type: application/json
                                                                                    # {
                                                                                    #      "message": "Hello, API!"
                                                                                    #   }
    return { #Returns a response with a success message and echoes back the message.
        "response": "Your request was successful! 🎉",
        "args": args["message"], #Extracts the value of the message key from the JSON payload (args["message"]).
    }


############################################################################################
#The /predict Endpoint
@app.route("/predict", methods=["POST"])
def run_predict() -> dict:
    import predict # import predict: Imports a module (predict.py) that handles prediction logic.

    try:
        args = flask.request.get_json() or {} # Retrieves the JSON payload from the client with the Required Inputs: bucket and data
        bucket = args["bucket"]
        model_dir = f"gs://{bucket}/model_output" # path to the model directory in Cloud Storage
        data = args["data"]
        predictions = predict.run(data, model_dir) #Calls the prediction function from the predict module:

        return {
            "method": "predict",
            "model_dir": model_dir,
            "predictions": predictions,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

# To Start the Web Server
# Flask creates a web server (development or production) that listens for requests on a specific IP address and port (e.g., http://localhost:8080).
if __name__ == "__main__":  #Ensures the app runs only when executed directly (not when imported as a module).
    import os

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    
    #Without starting the web server, your REST API endpoints (e.g., /ping or /predict) are not accessible.
