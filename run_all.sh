#!/bin/bash

GREEN="\033[1;32m"
NC="\033[0m"

cd ./src

for f in ./*.py; do
    echo -e "${GREEN}Running ${f}...${NC}"
    python ${f}
    echo -e "${GREEN}Finish${NC}"
done