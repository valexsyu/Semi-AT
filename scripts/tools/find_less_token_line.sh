#!/bin/bash

filename=$1
token_num=$2
row_number=0
while IFS= read -r line; do
    # Count the number of words in the line (assuming words are separated by spaces)
    word_count=$(echo "$line" | wc -w)
     ((row_number++))
    # Check if the word count is NOT equal to 200
    if [ "$word_count" -ne $token_num ]; then
        echo "the $row_number" is $word_count words >> "non_${token_num}_lines.txt"
    fi
done < "$filename"
