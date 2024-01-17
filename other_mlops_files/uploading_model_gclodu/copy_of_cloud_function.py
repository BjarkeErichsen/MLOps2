import functions_framework
from google.cloud import storage
import pickle
import sklearn

@functions_framework.http
def knn_endpoint(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """

    BUCKET_NAME = "cloud_deploy_bucket"
    MODEL_FILE = "model.pkl"

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(MODEL_FILE)
    my_model = pickle.loads(blob.download_as_string())

    def knn_classifier(request):
        """ will to stuff to your request """
        request_json = request.get_json()
        if request_json and 'input_data' in request_json:
            data = request_json['input_data']
            input_data = list(map(int, data.split(',')))
            prediction = my_model.predict([input_data])
            return f'Belongs to class: {prediction}'
        else:
            return 'No input data received'

    output = knn_classifier(request)
    return output
