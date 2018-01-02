from ct.cert_analysis import algorithm
from ct.cert_analysis import ca_field
from ct.cert_analysis import common_name
from ct.cert_analysis import crl_pointers
from ct.cert_analysis import dnsnames
from ct.cert_analysis import extensions
from ct.cert_analysis import ip_addresses
from ct.cert_analysis import ocsp_pointers
from ct.cert_analysis import serial_number
from ct.cert_analysis import validity

ALL_CHECKS = [serial_number.CheckNegativeSerialNumber(),
              validity.CheckValidityNotBeforeFuture(),
              validity.CheckValidityCorrupt(),
              validity.CheckIsExpirationDateWellDefined(),
              dnsnames.CheckValidityOfDnsnames(),
              dnsnames.CheckCorruptSANExtension(),
              dnsnames.CheckTldMatches(),
              common_name.CheckSCNTldMatches(),
              common_name.CheckLackOfSubjectCommonName(),
              common_name.CheckCorruptSubjectCommonName(),
              extensions.CheckCorrectExtensions(),
              ip_addresses.CheckPrivateIpAddresses(),
              ip_addresses.CheckCorruptIpAddresses(),
              algorithm.CheckSignatureAlgorithmsMismatch(),
              algorithm.CheckCertificateAlgorithmSHA1After2017(),
              algorithm.CheckTbsCertificateAlgorithmSHA1Ater2017(),
              ca_field.CheckCATrue(),
              ocsp_pointers.CheckOcspExistence(),
              ocsp_pointers.CheckCorruptOrMultipleAiaExtension(),
              crl_pointers.CheckCrlExistence(),
              crl_pointers.CheckCorruptOrMultipleCrlExtension(),]
