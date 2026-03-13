#!/bin/bash

if [ -z "$1" ]; then
    cd ./src
    python ./main.py
else
    echo "Signal-Bot currently unavailable."
fi