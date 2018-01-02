"""Metadata for Extended Validation certificates and roots"""

from collections import defaultdict

from ct.crypto.asn1 import oid

# Most root fingerprints (as well as EV policy OIDs) are from Chrome:
# https://code.google.com/p/chromium/codesearch#chromium/src/net/cert/ev_root_ca_metadata.cc
# In many cases the chains for EV certificates logged chain to an obsolete root
# which was replaced since logging.
# Fingerprints for these old roots (together with evidence of their
# origin) were added.
_EV_ROOTS = (
    # AC Camerfirma S.A. Chambers of Commerce Root - 2008
    # https://www.camerfirma.com
    (
        "786a74ac76ab147f9c6a3050ba9ea87efe9ace3c",
        # AC Camerfirma uses the last two arcs to track how the private key is
        # managed - the effective verification policy is the same.
        ("1.3.6.1.4.1.17326.10.14.2.1.2",
         "1.3.6.1.4.1.17326.10.14.2.2.2")
        ),

    # AC Camerfirma S.A. Global Chambersign Root - 2008
    # https://server2.camerfirma.com:8082
    (
        "4abdeeec950d359c89aec752a12c5b29f6d6aa0c",
        # AC Camerfirma uses the last two arcs to track how the private key is
        # managed - the effective verification policy is the same.
        ("1.3.6.1.4.1.17326.10.8.12.1.2",
         "1.3.6.1.4.1.17326.10.8.12.2.2")
        ),

    # AddTrust External CA Root
    # https://addtrustexternalcaroot-ev.comodoca.com
    (
        "02faf3e291435468607857694df5e45b68851868",
        ("1.3.6.1.4.1.6449.1.2.1.5.1",
         # This is the Network Solutions EV OID. However, this root
         # cross-certifies NetSol and so we need it here too.
         "1.3.6.1.4.1.782.1.2.1.8.1")
        ),

    # AffirmTrust Commercial
    # https://commercial.affirmtrust.com/
    (
        "f9b5b632455f9cbeec575f80dce96e2cc7b278b7",
        ("1.3.6.1.4.1.34697.2.1",),
        ),

    # AffirmTrust Networking
    # https://networking.affirmtrust.com:4431
    (
        "293621028b20ed02f566c532d1d6ed909f45002f",
        ("1.3.6.1.4.1.34697.2.2",),
        ),

    # AffirmTrust Premium
    # https://premium.affirmtrust.com:4432/
    (
        "d8a6332ce0036fb185f6634f7d6a066526322827",
        ("1.3.6.1.4.1.34697.2.3",),
        ),

    # AffirmTrust Premium ECC
    # https://premiumecc.affirmtrust.com:4433/
    (
        "b8236b002f1d16865301556c11a437caebffc3bb",
        ("1.3.6.1.4.1.34697.2.4",),
        ),

    # Buypass Class 3 CA 1
    # https://valid.evident.ca13.ssl.buypass.no/
    (
        "61573A11DF0ED87ED5926522EAD056D744B32371",
        ("2.16.578.1.26.1.3.3",),
        ),

    # Buypass Class 3 Root CA
    # https://valid.evident.ca23.ssl.buypass.no/
    (
        "DAFAF7FA6684EC068F1450BDC7C281A5BCA96457",
        ("2.16.578.1.26.1.3.3",),
        ),

    # CertPlus Class 2 Primary CA (KEYNECTIS)
    # https://www.keynectis.com/
    (
        "74207441729cdd92ec7931d823108dc28192e2bb",
        ("1.3.6.1.4.1.22234.2.5.2.3.1",),
        ),

    # Certum Trusted Network CA
    # https://juice.certum.pl/
    (
        "07e032e020b72c3f192f0628a2593a19a70f069e",
        ("1.2.616.1.113527.2.5.1.1",),
        ),

    # Certum CA
    # Was recognized by Microsoft at some point:
    # http://social.technet.microsoft.com/wiki/contents/articles/2592.windows-root-certificate-program-members-list-all-cas.aspx
     (
         "6252dc40f71143a22fde9ef7348e064251b18118",
         ("1.2.616.1.113527.2.5.1.1",),
         ),

    # China Internet Network Information Center EV Certificates Root
    # https://evdemo.cnnic.cn/
    (
        "4F99AA93FB2BD13726A1994ACE7FF005F2935D1E",
        ("1.3.6.1.4.1.29836.1.10",),
        ),

    # COMODO Certification Authority
    # https://secure.comodo.com/
    (
        "6631bf9ef74f9eb6c9d5a60cba6abed1f7bdef7b",
        ("1.3.6.1.4.1.6449.1.2.1.5.1",),
        ),

    # Some more Comodo, see
    # https://code.google.com/p/android/issues/detail?id=54479 and
    # https://code.google.com/p/chromium/issues/detail?id=231900
    (
        "afe5d244a8d1194230ff479fe2f897bbcd7a8cb4",
        ("1.3.6.1.4.1.6449.1.2.1.5.1",),
        ),
    (
        "2b8f1b57330dbba2d07a6c51f70ee90ddab9ad8e",
        ("1.3.6.1.4.1.6449.1.2.1.5.1",),
        ),
    (
        "d1cbca5db2d52a7f693b674de5f05a1d0c957df0",
        ("1.3.6.1.4.1.6449.1.2.1.5.1",),
        ),

    # COMODO Certification Authority (reissued certificate with NotBefore of Jan
    # 1 00:00:00 2011 GMT)
    # https://secure.comodo.com/
    (
        "ee869387fffd8349ab5ad14322588789a457b012",
        ("1.3.6.1.4.1.6449.1.2.1.5.1",),
        ),

    # COMODO ECC Certification Authority
    # https://comodoecccertificationauthority-ev.comodoca.com/
    (
        "9f744e9f2b4dbaec0f312c50b6563b8e2d93c311",
        ("1.3.6.1.4.1.6449.1.2.1.5.1",),
        ),

    # Cybertrust Global Root
    # https://evup.cybertrust.ne.jp/ctj-ev-upgrader/evseal.gif
    (
        "5f43e5b1bff8788cac1cc7ca4a9ac6222bcc34c6",
        ("1.3.6.1.4.1.6334.1.100.1",),
        ),

    # Baltimore CyberTrust Root
    (
        # See https://code.google.com/p/android/issues/detail?id=9269
         "d4de20d05e66fc53fe1a50882c78db2852cae474",
        # # Cybertrust Global Root
         ("1.3.6.1.4.1.6334.1.100.1",
          # DigiCert
          "2.16.840.1.114412.2.1"),
         ),

    # DigiCert High Assurance EV Root CA
    # https://www.digicert.com
    (
        "5fb7ee0633e259dbad0c4c9ae6d38f1a61c7dc25",
        ("2.16.840.1.114412.2.1",),
        ),

    # D-TRUST Root Class 3 CA 2 EV 2009
    # https://certdemo-ev-valid.ssl.d-trust.net/
    (
        "96c91b0b95b4109842fad0d82279fe60fab91683",
        ("1.3.6.1.4.1.4788.2.202.1",),
        ),

    # Entrust.net Secure Server Certification Authority
    # https://www.entrust.net/
    (
        "99a69be61afe886b4d2b82007cb854fc317e1539",
        ("2.16.840.1.114028.10.1.2",
        # DigiCert
        # "2.16.840.1.114412.2.1",
        # SecureTrust
        # "2.16.840.1.114404.1.1.2.4.1"
        ),
    ),

    # Entrust Root Certification Authority
    # https://www.entrust.net/
    (
        "b31eb1b740e36c8402dadc37d44df5d4674952f9",
        ("2.16.840.1.114028.10.1.2",),
        ),
    # Another Entrust root - Entrust Root Certification Authority - G2
    # Included in Mac OS X at the very least
    (
        "8cf427fd790c3ad166068de81e57efbb932272d4",
        ("2.16.840.1.114028.10.1.2",),
        ),
    # Old Entrust root
    # See https://bugzilla.mozilla.org/show_bug.cgi?id=849833
    (
        "801d62d07b449d5c5c035c98ea61fa443c2a58fe",
        ("2.16.840.1.114028.10.1.2",),
        ),

    # Entrust-DigiCert cross-signed intermediate
    # Entrust.net Certification Authority (2048)
     (
         # Replacement root, see
         # https://bugzilla.mozilla.org/show_bug.cgi?id=849833
         "503006091d97d4f5ae39f7cbe7927d7d652d3431",
         ("2.16.840.1.114412.2.1",),
         ),

    # Equifax Secure Certificate Authority (GeoTrust)
    # https://www.geotrust.com/
    (
        "d23209ad23d314232174e40d7f9d62139786633a",
        ("1.3.6.1.4.1.14370.1.6",),
        ),

    # E-Tugra Certification Authority
    # https://sslev.e-tugra.com.tr
    (
        "51C6E70849066EF392D45CA00D6DA3628FC35239",
        ("2.16.792.3.0.4.1.1.4",),
        ),

    # GeoTrust Primary Certification Authority
    # https://www.geotrust.com/
    (
        "323c118e1bf7b8b65254e2e2100dd6029037f096",
        ("1.3.6.1.4.1.14370.1.6",),
        ),

    # GeoTrust Primary Certification Authority - G2
    (
        "8d1784d537f3037dec70fe578b519a99e610d7b0",
        ("1.3.6.1.4.1.14370.1.6",),
        ),

    # GeoTrust Primary Certification Authority - G3
    (
        "039eedb80be7a03c6953893b20d2d9323a4c2afd",
        ("1.3.6.1.4.1.14370.1.6",),
        ),

    # GlobalSign Root CA - R2
    # https://www.globalsign.com/
    (
        "75e0abb6138512271c04f85fddde38e4b7242efe",
        ("1.3.6.1.4.1.4146.1.1",),
        ),

    # GlobalSign Root CA
    (
        "b1bc968bd4f49d622aa89a81f2150152a41d829c",
        ("1.3.6.1.4.1.4146.1.1",),
        ),

    # GlobalSign Root CA - R3
    # https://2029.globalsign.com/
    (
        "d69b561148f01c77c54578c10926df5b856976ad",
        ("1.3.6.1.4.1.4146.1.1",),
        ),
    # GlobalSign Root CA - phased out
    # See https://2014.globalsign.com/
    (
        "2f173f7de99667afa57af80aa2d1b12fac830338",
        ("1.3.6.1.4.1.4146.1.1",),
        ),


    # Go Daddy Class 2 Certification Authority
    # https://www.godaddy.com/
    (
        "2796bae63f1801e277261ba0d77770028f20eee4",
        ("2.16.840.1.114413.1.7.23.3",),
        ),

    # Go Daddy Root Certificate Authority - G2
    # https://valid.gdig2.catest.godaddy.com/
    (
        "47beabc922eae80e78783462a79f45c254fde68b",
        ("2.16.840.1.114413.1.7.23.3",),
        ),

    # GTE CyberTrust Global Root
    # https://www.cybertrust.ne.jp/
    (
        "97817950d81c9670cc34d809cf794431367ef474",
        ("1.3.6.1.4.1.6334.1.100.1",
         # DigiCert
         # "2.16.840.1.114412.2.1"
         ),
        ),
    # Izenpe.com - SHA256 root
    # The first OID is for businesses and the second for government entities.
    # These are the test sites, respectively:
    # https://servicios.izenpe.com
    # https://servicios1.izenpe.com
    (
        "2f783d255218a74a653971b52ca29c45156fe919",
    ("1.3.6.1.4.1.14777.6.1.1", "1.3.6.1.4.1.14777.6.1.2"),
    ),

    # Izenpe.com - SHA1 root
    # Windows XP finds this, SHA1, root instead. The policy OIDs are the same as
    # for the SHA256 root, above.
    (
        "30779e9315022e94856a3ff8bcf815b082f9aefd",
        ("1.3.6.1.4.1.14777.6.1.1", "1.3.6.1.4.1.14777.6.1.2"),
        ),

    # Network Solutions Certificate Authority
    # https://www.networksolutions.com/website-packages/index.jsp
    (
        "74f8a3c3efe7b390064b83903c21646020e5dfce",
        ("1.3.6.1.4.1.782.1.2.1.8.1",),
        ),

    # Network Solutions Certificate Authority (reissued certificate with
    # NotBefore of Jan  1 00:00:00 2011 GMT).
    # https://www.networksolutions.com/website-packages/index.jsp
    (
        "71899a67bf33af31befdc071f8f733b183856332",
        ("1.3.6.1.4.1.782.1.2.1.8.1",),
        ),

    # QuoVadis Root CA 2
    # https://www.quovadis.bm/
    (
        "ca3afbcf1240364b44b216208880483919937cf7",
        ("1.3.6.1.4.1.8024.0.2.100.1.2",),
        ),

    # QuoVadis Root (cross-signing QuoVadis Root CA 2)
    # https://www.quovadis.bm/
    # Old certificate, see:
    # http://www.quovadisglobal.co.uk/~/media/Files/Repository/QV_RCA1_RCA3_CPCPS_V4_5.ashx
     (
         "de3f40bd5093d39b6c60f6dabc076201008976c9",
         ("1.3.6.1.4.1.8024.0.2.100.1.2",),
         ),


    # SecureTrust CA, SecureTrust Corporation
    # https://www.securetrust.com
    # https://www.trustwave.com/
    (
        "8782c6c304353bcfd29692d2593e7d44d934ff11",
        ("2.16.840.1.114404.1.1.2.4.1",),
        ),

    # Secure Global CA, SecureTrust Corporation
    (
        "3a44735ae581901f248661461e3b9cc45ff53a1b",
        ("2.16.840.1.114404.1.1.2.4.1",),
        ),

    # Security Communication RootCA1
    # https://www.secomtrust.net/contact/form.html
    (
        "36b12b49f9819ed74c9ebc380fc6568f5dacb2f7",
        ("1.2.392.200091.100.721.1",),
        ),

    # Security Communication EV RootCA1
    # https://www.secomtrust.net/contact/form.html
    (
        "feb8c432dcf9769aceae3dd8908ffd288665647d",
        ("1.2.392.200091.100.721.1",),
        ),

    # StartCom Certification Authority
    # https://www.startssl.com/
    (
        "3e2bf7f2031b96f38ce6c4d8a85d3e2d58476a0f",
        ("1.3.6.1.4.1.23223.1.1.1",),
        ),

    # StartCom Certification Authority G2
    # https://www.startssl.com/
    # Renewed StartCom root certs, see:
    # https://bugzilla.mozilla.org/show_bug.cgi?id=751954
     (
         "31f1fd68226320eec63b3f9dea4a3e537c7c3917",
         ("1.3.6.1.4.1.23223.1.1.1",),
         ),

    # StartCom Certification Authority (SHA2)
    # https://www.startssl.com/
    # Renewed StartCom root certs, see:
    # https://bugzilla.mozilla.org/show_bug.cgi?id=751954
     (
         "a3f1333fe242bfcfc5d14e8f394298406810d1a0",
         ("1.3.6.1.4.1.23223.1.1.1",),
         ),

    # Starfield Class 2 Certification Authority
    # https://www.starfieldtech.com/
    (
        "ad7e1c28b064ef8f6003402014c3d0e3370eb58a",
        ("2.16.840.1.114414.1.7.23.3",),
        ),

    # Starfield Root Certificate Authority - G2
    # https://valid.sfig2.catest.starfieldtech.com/
    (
        "b51c067cee2b0c3df855ab2d92f4fe39d4e70f0e",
        ("2.16.840.1.114414.1.7.23.3",),
        ),

    # Starfield Services Root Certificate Authority - G2
    # https://valid.sfsg2.catest.starfieldtech.com/
    (
        "925a8f8d2c6d04e0665f596aff22d863e8256f3f",
        ("2.16.840.1.114414.1.7.24.3",),
        ),

    # SwissSign Gold CA - G2
    # https://testevg2.swisssign.net/
    (
        "d8c5388ab7301b1b6ed47ae645253a6f9f1a2761",
        ("2.16.756.1.89.1.2.1.1",
         # Affirmtrust cross-sign
         # "1.3.6.1.4.1.34697.2.2"
         ),
        ),

    # Thawte Premium Server CA
    # https://www.thawte.com/
    (
        "627f8d7827656399d27d7f9044c9feb3f33efa9a",
        ("2.16.840.1.113733.1.7.48.1",),
        ),

    # thawte Primary Root CA
    # https://www.thawte.com/
    (
        "91c6d6ee3e8ac86384e548c299295c756c817b81",
        ("2.16.840.1.113733.1.7.48.1",),
        ),

    # thawte Primary Root CA - G2
    (
        "aadbbc22238fc401a127bb38ddf41ddb089ef012",
        ("2.16.840.1.113733.1.7.48.1",),
        ),

    # thawte Primary Root CA - G3
    (
        "f18b538d1be903b6a6f056435b171589caf36bf2",
        ("2.16.840.1.113733.1.7.48.1",),
        ),
    # Thawte Root 2
    # Thawte Premium Server CA - http://www.thawte.com/roots/
    # Transition certificate, see Root 2 under http://www.thawte.com/roots/
    (
        "e0ab059420725493056062023670f7cd2efc6666",
        ("2.16.840.1.113733.1.7.48.1",),
        ),

    # TWCA Global Root CA
    # https://evssldemo3.twca.com.tw/index.html
    (
        "9CBB4853F6A4F6D352A4E83252556013F5ADAF65",
        ("1.3.6.1.4.1.40869.1.1.22.3",),
        ),

    # TWCA Root Certification Authority
    # https://evssldemo.twca.com.tw/index.html
    (
        "cf9e876dd3ebfc422697a3b5a37aa076a9062348",
        ("1.3.6.1.4.1.40869.1.1.22.3",),
        ),

    # T-TeleSec GlobalRoot Class 3
    # http://www.telesec.de/ / https://root-class3.test.telesec.de/
    (
        "55a6723ecbf2eccdc3237470199d2abe11e381d1",
        ("1.3.6.1.4.1.7879.13.24.1",),
        ),

    # UTN - DATACorp SGC
    (
        "58119f0e128287ea50fdd987456f4f78dcfad6d4",
        ("1.3.6.1.4.1.6449.1.2.1.5.1",),
        ),

    # UTN-USERFirst-Hardware
    (
        "0483ed3399ac3608058722edbc5e4600e3bef9d7",
        ("1.3.6.1.4.1.6449.1.2.1.5.1",
         # This is the Network Solutions EV OID. However, this root
         # cross-certifies NetSol and so we need it here too.
         "1.3.6.1.4.1.782.1.2.1.8.1"),
        ),

    # ValiCert Class 2 Policy Validation Authority
    (
        "317a2ad07f2b335ef5a1c34e4b57e8b7d8f1fca6",
        ("2.16.840.1.114413.1.7.23.3", "2.16.840.1.114414.1.7.23.3"),
        ),

    # ValiCert Class 1 Policy Validation Authority
    # Old 1024 bit root, see
    # https://developer.mozilla.org/en-US/docs/Mozilla/Projects/NSS/NSS_3.16.3_release_notes
    (
        "e5df743cb601c49b9843dcab8ce86a81109fe48e",
        # SECOM
        ("1.2.392.200091.100.721.1",),
        ),

    # VeriSign Class 3 Public Primary Certification Authority (MD2)
    # https://www.verisign.com/
    (
        "742c3192e607e424eb4549542be1bbc53e6174e2",
        ("2.16.840.1.113733.1.7.23.6",),
        ),

    # VeriSign Class 3 Public Primary Certification Authority (SHA1)
    # https://www.verisign.com/
    # Legacy certificate, see Root 2 under
    # http://www.symantec.com/page.jsp?id=roots
    (
        "a1db6393916f17e4185509400415c70240b0ae6b",
        ("2.16.840.1.113733.1.7.23.6",),
        ),


    # VeriSign Class 3 Public Primary Certification Authority - G4
    (
        "22D5D8DF8F0231D18DF79DB7CF8A2D64C93F6C3A",
        ("2.16.840.1.113733.1.7.23.6",),
        ),

    # VeriSign Class 3 Public Primary Certification Authority - G5
    # https://www.verisign.com/
    (
        "4eb6d578499b1ccf5f581ead56be3d9b6744a5e5",
        ("2.16.840.1.113733.1.7.23.6",),
        ),

    # VeriSign Universal Root Certification Authority
    (
        "3679ca35668772304d30a5fb873b0fa77bb70d54",
        ("2.16.840.1.113733.1.7.23.6",),
        ),

    # Wells Fargo WellsSecure Public Root Certificate Authority
    # https://nerys.wellsfargo.com/test.html
    (
        "e7b4f69d61ec9069db7e90a7401a3cf47d4fe8ee",
        ("2.16.840.1.114171.500.9",),
        ),

    # XRamp Global Certification Authority
    (
        "b80186d1eb9c86a54104cf3054f34c52b7e558c6",
        ("2.16.840.1.114404.1.1.2.4.1",),
        ),

    # Pending inclusion in Chrome
    # Swisscom Root EV CA 2
    (
        "e7a19029d3d552dc0d0fc692d3ea880d152e1a6b",
        ("2.16.756.1.83.21.0",),
        ),
    # Buypass Class 3 Root CA
    (
        "dafaf7fa6684ec068f1450bdc7c281a5bca96457",
        ("2.16.578.1.26.1.3.3",),
        ),
    # Autoridad de Certificacion Firmaprofesional CIF A62634068
    (
        "aec5fb3fc8e1bfc4e54f03075a9ae800b7f7b6fa",
        ("1.3.6.1.4.1.13177.10.1.3.10",),
        ),
    # TURKTRUST Elektronik Sertifika Hizmet Saglayicisi
    (
        "f17f6fb631dc99e3a3c87ffe1cf1811088d96033",
        ("2.16.792.3.0.3.1.1.5",),
        ),
    # Actalis Authentication Root CA
    (
        "f373b387065a28848af2f34ace192bddc78e9cac",
        ("1.3.159.1.17.1",),
        ),
    # QuoVadis Root CA 2 G3
    (
        "093c61f38b8bdc7d55df7538020500e125f5c836",
        ("1.3.6.1.4.1.8024.0.2.100.1.2",),
        )
)
####### Added #######
## Entrust - DigiCert x2
## GTE CyberTrust - Digicert
## VeriSign SHA1
## StartCom SHA2, StartCom G2
## SwissSign - AffirmTrust (vice versa: chain terminating at AffirmTrust
##                          includes SwissSign policy).
## QuoVadis - QuoVadis (EV on FF and not on Chrome)
## Entrust - SecureTrust
## Certum CA - Certum Trusted Network CA
## Valicert Class 1 - SECOM
## Baltimore Cybertrust Root - Cybertrust Global Root, DigiCert
## Swisscom Root EV CA 2
## Buypass Class 3 Root CA
## Autoridad de Certificacion Firmaprofesional CIF A62634068
## TURKTRUST Elektronik Sertifika Hizmet Saglayicisi
## Actalis Authentication Root CA
## QuoVadis Root CA 2 G3

EV_POLICIES = defaultdict(set)
EV_ROOTS = defaultdict(set)

for chash, policies in _EV_ROOTS:
    chash_raw = chash.decode("hex")
    assert len(chash_raw) == 20
    for policy in policies:
        EV_POLICIES[oid.ObjectIdentifier(value=policy)].add(chash_raw)
        EV_ROOTS[chash_raw].add(policy)
