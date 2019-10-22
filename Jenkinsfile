pipeline {
    environment {
        registry = 'kmitsunami/cicdpipeline'
        registryCredential = 'dockerhub'
    }
    agent any
    stages {
        stage('Lint') {
            steps {
                sh 'tidy -q -e src/*.html'
                sh 'pylint --disable=R,C,W1203,W0105,W0110,W0212,E0401 app.py'
                sh 'hadolint Dockerfile'
            }
        }
        stage('Build Docker') {
            steps {
                script {
                    dockerImage = SUDO docker.build(registry + ":$BUILD_NUMBER")
                }
            }
        }
        stage('Push Docker') {
            steps {
                script {
                    docker.withRegistry('', registryCredential){
                        dockerImage.push()
                    }
                }
            }
        }
    }
}
