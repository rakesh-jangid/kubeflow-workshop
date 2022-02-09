while ! kustomize build kubeflow1.3 | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
