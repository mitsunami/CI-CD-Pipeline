pipeline {
    agent any
    stages {
        stage('Lint HTML') {
            steps {
                sh 'tidy -q -e src/*.html'
            }
        }
        stage('Upload to AWS') {
            steps {
                sh 'echo "Hello World"'
                sh '''
                    echo "Multiline shell steps works too"
                    ls -lah
                '''
                withAWS(credentials: 'aws-static') {
                    s3Upload(file:'index.html', bucket:'udacity-jenkins-project', path: 'index.html', verbose:true)
                }
            }
        }
    }
}
