#!/bin/bash

# this just takes php's date() function as a reference to check if week of year
# is calculated correctly in the range from 1970 .. 2038 by brute force...

SEQ="seq"
SYSTEM=`uname`
if [ "$SYSTEM" = "Darwin" ]; then
	SEQ="jot"
fi

for YEAR in {1970..2038}; do
  for MONTH in {1..12}; do
    DAYS=$(cal $MONTH $YEAR | egrep "28|29|30|31" |tail -1 |awk '{print $NF}')
    for DAY in $( $SEQ $DAYS ); do
      DATE=$YEAR-$MONTH-$DAY
      echo -n $DATE ...
      NODEVAL=$(node test_weekofyear.js $DATE)
      PHPVAL=$(php -r "echo intval(date('W', strtotime('$DATE')));")
      if [ "$NODEVAL" -ne "$PHPVAL" ]; then
        echo "MISMATCH: node: $NODEVAL vs php: $PHPVAL for date $DATE"
      else
        echo " OK"
      fi
    done
  done
done
