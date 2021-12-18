# Containerizing an Application using Docker and ECR

1. Create simple app
2. Create Dockerfile
3. Build image
4. Create ECR repo
5. Push image to the repo


# Simple Python app 

Create src dir
```
mkdir src
```

src/server.py
```
from http.server import HTTPServer, BaseHTTPRequestHandler


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')


print("starting server...")
httpd = HTTPServer(('0.0.0.0', 8001), SimpleHTTPRequestHandler)
httpd.serve_forever()

```

Let's run it:
```
python server.py
```

and test it: open web browser on http://localhost:8001/. You should see 'Hello, world!'

Type ctrl-c in terminal to shut down server.py

# Dockerfile

Dockerfile
```
FROM python:3.8-slim-buster
ENV PYTHONUNBUFFERED=1
ENV SRC_DIR /usr/bin/src/webapp/src
COPY src/* ${SRC_DIR}/
WORKDIR ${SRC_DIR}
EXPOSE 8001
CMD ["python", "server.py"]
```

To build our first image type:
```
docker build . -t simple_server
```

List images with:

```
docker images
```

Now you should see simple_server in the list.

Let's run our docker image(and make sure port 8001 is free on local your machine):

```
docker run -p 8001:8001 -d simple_server
```

If everythin goes right you should see it running:

```
docker ps
```

Ok, let's kill it, using container id take from docker ps command
```
docker kill <container_id>
```

# ECR Repository CloudFormation script

ECRRepo.yaml
```
AWSTemplateFormatVersion: "2010-09-09"
Description: "ECR"
Parameters:
  ECRRepoName:
    Type: String
Resources:
  ECRRepository:
    Type: AWS::ECR::Repository
    Properties:
      RepositoryName: !Ref ECRRepoName
      ImageScanningConfiguration:
        scanOnPush: "true"
      Tags:
        - Key: Name
          Value: !Ref ECRRepoName

```

Deploy repository from cmd line (or using AWS console)

```
aws cloudformation create-stack --stack-name ECRR1 --template-body file://ECRRepo.yaml --parameters ParameterKey=ECRRepoName,ParameterValue=repo1
```

Let's login to the repo.
Substitue <account_id> with your account id and try to get login:

```
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-west-2.amazonaws.com
```
You shoudl see "Login Succeeded"

Now, let's tag out simple_server image with ecr tag:
```
docker tag simple_server:latest <account_id>.dkr.ecr.us-west-2.amazonaws.com/repo1:latest
```

Then, check if image was properly tagged:
```
docker images
```

And then push docker image into AWS ECR repository:
```
docker push <account_id>.dkr.ecr.us-west-2.amazonaws.com/repo1:latest
```

Now let's take a look at AWS console -> ECR. Check if images is there, take a look at Vulnarabilites tab...
