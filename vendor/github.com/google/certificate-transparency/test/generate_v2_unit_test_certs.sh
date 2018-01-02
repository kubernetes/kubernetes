# This generates a bunch of certs for unit tests of validating the rules
# sections 3.2.2 and 3.2.3 of the V2 CT RFC. They are intended for use
# in tests only. Some of the output certs do not follow the rules in the RFC.
# Test case numbering relates to spreadsheet. Not all cases may be tested so
# there could be gaps in numbering.
#
# NOTE: Our Git repo includes the test certs created by this script. You don't
# need to run it again unless you wish to regenerate them for some reason.

set -e

TMP_REQ=$(mktemp) || { echo "Could not create temp request file"; exit 1; }

create_cert() {
  config=$1
  ext_section=$2
  pem_file=$3

  rm -f $pem_file

  openssl req -new -key testdata/ca-key.pem -out ${TMP_REQ} \
    -config $config -passin pass:password1
  openssl x509 -req -days 365 -in ${TMP_REQ} -CA testdata/ca-cert.pem \
    -CAkey testdata/ca-key.pem -passin pass:password1 -CAcreateserial \
    -out $pem_file -extfile $config -extensions $ext_section
} 

make_redaction_test_case() {
  test_name=$1

  echo $1 ':'

  create_cert testdata/v2/configs/redact/${test_name}.config $1 \
    testdata/v2/redact_${test_name}.pem
}

make_name_constraint_test_case() {
  test_name=$1

  echo $1 ':'

  create_cert testdata/v2/configs/constraint/${test_name}.config $1 \
    testdata/v2/constraint_${test_name}.pem
}

# Generate the redaction test cases

make_redaction_test_case test5
make_redaction_test_case test6
make_redaction_test_case test7
make_redaction_test_case test8
make_redaction_test_case test9
make_redaction_test_case test10
make_redaction_test_case test11
make_redaction_test_case test12
make_redaction_test_case test13
make_redaction_test_case test14
make_redaction_test_case test15
make_redaction_test_case test22
make_redaction_test_case test23
make_redaction_test_case test24

# Generate the name constraint test cases

make_name_constraint_test_case test2
make_name_constraint_test_case test3
make_name_constraint_test_case test4
make_name_constraint_test_case test5
make_name_constraint_test_case test6
make_name_constraint_test_case test7
make_name_constraint_test_case test8
make_name_constraint_test_case test9

