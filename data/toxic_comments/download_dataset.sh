#!/usr/bin/env bash

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
cp ~/.kaggle/competitions/jigsaw-toxic-comment-classification-challenge/*.zip .

for f in *.zip
do
	unzip $f
	rm $f
done
