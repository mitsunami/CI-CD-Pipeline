#!/usr/bin/env bash

eksctl create cluster --name cicdpipeline   --region us-west-2 --nodes 2 --nodes-min 1 --nodes-max 2 --node-type t2.medium 
kubectl apply -f amazon_eks/deployment.yml
kubectl apply -f amazon_eks/services.yml
