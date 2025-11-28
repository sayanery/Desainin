from app import app
from serverless_wsgi import handle_request

def main(request, response):
    return handle_request(app, request, response)
