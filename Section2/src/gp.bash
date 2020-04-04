#!/usr/bin/env bash
if [ $1 == 1 ]; then
    javac -cp epochx-1.4.1.jar:commons-lang-2.5.jar:. Control.java 
    java -cp epochx-1.4.1.jar:commons-lang-2.5.jar:. Control.java training.arff test.arff
else
    javac -cp epochx-1.4.1.jar:commons-lang-2.5.jar:. Control.java 
    java -cp epochx-1.4.1.jar:commons-lang-2.5.jar:. Control.java training.arff test.arff | tail -n +6 > $1
    (echo "Generation	Loss" && cat $1) > tmp 
    mv tmp $1
fi
