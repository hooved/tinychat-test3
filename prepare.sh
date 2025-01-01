#!/usr/bin/env bash

# TODO: activate venv, run tinygrad/examples/tinychat/compile.py and make_tiktoken_js.sh
SOURCE_DIR=./tinygrad/examples/tinychat
cp ${SOURCE_DIR}/{index.html,index.js,index.css,net.js,tiktoken.js,common.css,favicon.svg} ./
cp -r $SOURCE_DIR/assets ./