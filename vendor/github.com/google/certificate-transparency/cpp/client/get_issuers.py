#!/usr/bin/env python

# Get the issuers for the supplied certificates from the AIA
# extension. Note that this is not very clever and may supply
# intermediates that are not needed and fail to supply intermediates
# that are needed.

# Based on pyasn1 example code.

from base64 import b64encode
from pyasn1.codec.der import decoder
from pyasn1.codec.der import encoder
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import pem
from pyasn1_modules import rfc2459
import sys
from urllib2 import urlopen

if len(sys.argv) != 1:
  print """Usage:
  $ %s < somecertificates.pem""" % sys.argv[0]
  sys.exit(-1)

cStart = '-----BEGIN CERTIFICATE-----'
cEnd = '-----END CERTIFICATE-----'

certType = rfc2459.Certificate()

# RFC 2459 is not sufficient for X509v3 certificates, extra stuff here.
# RFC 5280 4.2.2.1

id_pe_authorityInfoAccess = univ.ObjectIdentifier('1.3.6.1.5.5.7.1.1')

class AccessDescription(univ.Sequence):
  """
     AccessDescription  ::=  SEQUENCE {
                accessMethod          OBJECT IDENTIFIER,
                accessLocation        GeneralName  }
  """
  componentType = namedtype.NamedTypes(
    namedtype.NamedType('accessMethod', univ.ObjectIdentifier()),
    namedtype.NamedType('accessLocation', rfc2459.GeneralName()))

class AuthorityInfoAccessSyntax(univ.SequenceOf):
  """
  AuthorityInfoAccessSyntax  ::=
             SEQUENCE SIZE (1..MAX) OF AccessDescription
  """
  # FIXME: SIZE not encoded.
  componentType = AccessDescription()

id_ad_caIssuers = univ.ObjectIdentifier('1.3.6.1.5.5.7.48.2')

# End of RFC 5280 4.2.2.1

certCnt = 0

while 1:
  idx, substrate = pem.readPemBlocksFromFile(
    sys.stdin, (cStart, cEnd)
    )
  if not substrate:
    break

  cert, rest = decoder.decode(substrate, asn1Spec=certType)

  if rest: substrate = substrate[:-len(rest)]

  print cert.prettyPrint()

  tbs = cert.getComponentByName('tbsCertificate')
  extensions = tbs.getComponentByName('extensions') or []

  for extension in extensions:
    oid = extension.getComponentByName('extnID')
    if oid != id_pe_authorityInfoAccess:
      continue
    
    print extension.prettyPrint()

    value, rest = decoder.decode(extension.getComponentByName('extnValue'),
                                 asn1Spec=univ.OctetString())
    assert rest == ""
    aia, rest = decoder.decode(value, asn1Spec=AuthorityInfoAccessSyntax())
    assert rest == ""

    print aia.prettyPrint()

    for ad in aia:
      oid = ad.getComponentByName('accessMethod')
      if oid != id_ad_caIssuers:
        continue
      
      print ad.prettyPrint()

      loc = ad.getComponentByName('accessLocation').\
        getComponentByName('uniformResourceIdentifier')
      print type(loc), loc

      certHandle = urlopen(str(loc))
      cert = certHandle.read()
      print cStart
      b64 = b64encode(cert)
      for n in range(0, len(b64), 64):
        print b64[n:n+64]
      print cEnd
    
  certCnt = certCnt + 1

print('*** %s PEM cert(s) de/serialized' % certCnt)
