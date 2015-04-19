SAUCE_ACCESS_KEY=`echo $SAUCE_ACCESS_KEY | rev`

if [ $JOB = "smoke" ]; then
  node bin/protractor spec/smokeConf.js
elif [ $JOB = "suite" ]; then
  node bin/protractor spec/ciConf.js
else
  echo "Unknown job type. Please set JOB=smoke or JOB=suite"
fi
