#!/bin/bash
### Copyright 2010 Manuel Carrasco Moñino. (manolo at apache.org)
###
### Licensed under the Apache License, Version 2.0.
### You may obtain a copy of it at
### http://www.apache.org/licenses/LICENSE-2.0

###
### A library for shell scripts which creates reports in jUnit format.
### These reports can be used in Jenkins, or any other CI.
###
### Usage:
###     - Include this file in your shell script
###     - Use juLog to call your command any time you want to produce a new report
###        Usage:   juLog <options> command arguments
###           options:
###             -class="MyClass" : a class name which will be shown in the junit report
###             -name="TestName" : the test name which will be shown in the junit report
###             -error="RegExp"  : a regexp which sets the test as failure when the output matches it
###             -ierror="RegExp" : same as -error but case insensitive
###             -output="Path"   : path to output directory, defaults to "./results"
###     - Junit reports are left in the folder 'result' under the directory where the script is executed.
###     - Configure Jenkins to parse junit files from the generated folder
###

asserts=00; errors=0; total=0; content=""
date=`which gdate 2>/dev/null || which date`

# default output folder
juDIR=`pwd`/results

# The name of the suite is calculated based in your script name
suite=""

# A wrapper for the eval method witch allows catching seg-faults and use tee
errfile=/tmp/evErr.$$.log
function eVal() {
  (eval "$1")
  # stdout and stderr may currently be inverted (see below) so echo may write to stderr
  echo $? 2>&1 | tr -d "\n" > $errfile
}

# Method to clean old tests
function juLogClean() {
  echo "+++ Removing old junit reports from: $juDIR "
  rm -f "$juDIR"/junit-*
}

# Execute a command and record its results
function juLog() {
  suite="";
  errfile=/tmp/evErr.$$.log
  date=`which gdate || which date`
  asserts=00; errors=0; total=0; content=""

  # parse arguments
  ya=""; icase=""
  while [ -z "$ya" ]; do
    case "$1" in
  	  -name=*)   name=`echo "$1" | sed -e 's/-name=//'`;   shift;;
  	  -class=*)  class=`echo "$1" | sed -e 's/-class=//'`;   shift;;
      -ierror=*) ereg=`echo "$1" | sed -e 's/-ierror=//'`; icase="-i"; shift;;
      -error=*)  ereg=`echo "$1" | sed -e 's/-error=//'`;  shift;;
      -output=*) juDIR=`echo "$1" | sed -e 's/-output=//'`;  shift;;
      *)         ya=1;;
    esac
  done

  # create output directory
  mkdir -p "$juDIR" || exit
  # use first arg as name if it was not given
  if [ -z "$name" ]; then
    name="$asserts-$1"
    shift
  fi

  if [[ "$class" = "" ]]; then
    class="default"
  fi

  suite=$class

  # calculate command to eval
  [ -z "$1" ] && return
  cmd="$1"; shift
  while [ -n "$1" ]
  do
     cmd="$cmd \"$1\""
     shift
  done

  # eval the command sending output to a file
  outf=/var/tmp/ju$$.txt
  errf=/var/tmp/ju$$-err.txt
  >$outf
  echo ""                         | tee -a $outf
  echo "+++ Running case: $class.$name " | tee -a $outf
  echo "+++ working dir: "`pwd`           | tee -a $outf
  echo "+++ command: $cmd"            | tee -a $outf
  ini=`$date +%s.%N`
  # execute the command, temporarily swapping stderr and stdout so they can be tee'd to separate files,
  # then swapping them back again so that the streams are written correctly for the invoking process
  ( (eVal "$cmd" | tee -a $outf) 3>&1 1>&2 2>&3 | tee $errf) 3>&1 1>&2 2>&3
  evErr=`cat $errfile`
  rm -f $errfile
  end=`$date +%s.%N`
  echo "+++ exit code: $evErr"        | tee -a $outf

  # set the appropriate error, based in the exit code and the regex
  [ $evErr != 0 ] && err=1 || err=0
  out=`cat $outf | sed -e 's/^\([^+]\)/| \1/g'`
  if [ $err = 0 -a -n "$ereg" ]; then
      H=`echo "$out" | egrep $icase "$ereg"`
      [ -n "$H" ] && err=1
  fi
  echo "+++ error: $err"         | tee -a $outf
  rm -f $outf

  errMsg=`cat $errf`
  rm -f $errf
  # calculate vars
  asserts=$(($asserts+1))
  errors=$(($errors+$err))
  time=`echo "$end - $ini" | bc -l`
  total=`echo "$total + $time" | bc -l`

  # write the junit xml report
  ## failure tag
  [ $err = 0 ] && failure="" || failure="
      <failure type=\"ScriptError\" message=\"Script Error\">
      <![CDATA[
      $errMsg
      ]]>
      </failure>
  "
  ## testcase tag
  content="$content
    <testcase assertions=\"1\" name=\"$name\" time=\"$time\" classname=\"$class\">
    $failure
    <system-out>
<![CDATA[
$out
]]>
    </system-out>
    <system-err>
<![CDATA[
$errMsg
]]>
    </system-err>
    </testcase>
  "
  ## testsuite block

  if [[ -e "$juDIR/junit-$suite.xml" ]]; then
    # file exists. first update the failures count
    failCount=`sed -n "s/.*testsuite.*failures=\"\([0-9]*\)\".*/\1/p" "$juDIR/junit-$suite.xml"`
    errors=$(($failCount+$errors))
    sed -i "0,/failures=\"$failCount\"/ s/failures=\"$failCount\"/failures=\"$errors\"/" "$juDIR/junit-$suite.xml"
    sed -i "0,/errors=\"$failCount\"/ s/errors=\"$failCount\"/errors=\"$errors\"/" "$juDIR/junit-$suite.xml"

    # file exists. Need to append to it. If we remove the testsuite end tag, we can just add it in after.
    sed -i "s^</testsuite>^^g" $juDIR/junit-$suite.xml ## remove testSuite so we can add it later
    cat <<EOF >> "$juDIR/junit-$suite.xml"
     $content
    </testsuite>
EOF

  else
    # no file exists. Adding a new file
    cat <<EOF > "$juDIR/junit-$suite.xml"
    <testsuite failures="$errors" assertions="$assertions" name="$suite" tests="1" errors="$errors" time="$total">
    $content
    </testsuite>
EOF
  fi

}