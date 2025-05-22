#!/bin/bash

set -ex

clang++ -g -Wall -std=c++20 parser.cpp -o parser
