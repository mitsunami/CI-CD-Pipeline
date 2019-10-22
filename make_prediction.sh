#!/usr/bin/env bash

PORT=8000
echo "Port: $PORT"

# POST method predict
curl -d '{  
   "IMAGE":{  
      "0":1
   }
}'\
     -H "Content-Type: application/json" \
     -X POST http://localhost:$PORT/predict
