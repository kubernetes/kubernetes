import sys
from m2ext import SSL
from M2Crypto import X509

print "Validating certificate %s using CApath %s" % (sys.argv[1], sys.argv[2])
cert = X509.load_cert(sys.argv[1])
ctx = SSL.Context()
ctx.load_verify_locations(capath=sys.argv[2])
if ctx.validate_certificate(cert):
    print "valid"
else:
    print "invalid"
