#!/bin/bash -e


# Sed to process GitHub Markdown
# 1-2 Remove comment code from metadata block
# 
for i in ls -l /docs/content/*
  do                 # Line breaks are important
    if [ -d $i ]   # Spaces are important
      then
        y=${i##*/}
        find $i -type f -name "*.md" -exec sed -i.old \
        -e '/^<!.*metadata]>/g' \
        -e '/^<!.*end-metadata.*>/g' {} \;
    fi
done




