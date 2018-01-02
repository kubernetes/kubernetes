import base64
import dns.resolver
import dns.rdatatype
import logging
import math
import os
import random
import shlex
import signal
import subprocess
import sys
import time

NUMBER_OF_CERTS = 100

basepath = os.path.dirname(sys.argv[0])

sys.path.append(os.path.join(basepath, '../../python'))
from ct.crypto import merkle
from ct.proto import ct_pb2

tmpdir = sys.argv[1]

class CTDNSLookup:
    def __init__(self, nameservers, port):
        self.resolver = dns.resolver.Resolver(configure=False)
        self.resolver.nameservers = nameservers
        self.resolver.port = port

    def Get(self, name):
        answers = self.resolver.query(name, 'TXT')
        assert answers.rdtype == dns.rdatatype.TXT
        return answers

    def GetOne(self, name):
        answers = self.Get(name)
        assert len(answers) == 1
        txt = answers[0]
        assert len(txt.strings) == 1
        return txt.strings[0]

    def GetSTH(self):
        sth_str = self.GetOne('sth.example.com')
        sth = ct_pb2.SignedTreeHead()
        parts = str(sth_str).split('.')
        sth.tree_size = int(parts[0])
        sth.timestamp = int(parts[1])
        sth.sha256_root_hash = base64.b64decode(parts[2])
        #FIXME(benl): decompose signature into its parts
        #sth.signature = base64.b64decode(parts[3])
        return sth

    def GetEntry(self, level, index, size):
        return self.GetOne(str(level) + '.' + str(index) + '.' + str(size)
                           + '.tree.example.com')

    def GetLeafHash(self, index):
        return self.GetOne(str(index) + '.leafhash.example.com')

class DNSServerRunner:
    def Run(self, cmd):
        args = shlex.split(cmd)
        self.proc = subprocess.Popen(args)

def OpenSSL(*params):
    logging.info("RUN: openssl " + str(params))
    null = open("/dev/null")
    subprocess.check_call(("openssl",) + params, stdout=null, stderr=null)

class timeout:
    def __init__(self, seconds, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class CTServer:
    def __init__(self, cmd, base, ca):
        self.cmd_ = cmd
        self.base_ = base
        self.ca_ = ca
        self.GenerateKey()

    def __del__(self):
        self.proc.terminate()
        self.proc.wait()

    def PrivateKey(self):
        return self.base_ + "-ct-server-private-key.pem"
        
    def PublicKey(self):
        return self.base_ + "-ct-server-public-key.pem"

    def Database(self):
        return self.base_ + "-database.sqlite"

    def GenerateKey(self):
        OpenSSL("ecparam",
                "-out", self.PrivateKey(),
                "-name", "secp256r1",
                "-genkey")
        OpenSSL("ec",
                "-in", self.PrivateKey(),
                "-pubout",
                "-out", self.PublicKey())

    def URL(self):
        return "http://localhost:9999/"

    def Run(self):
        cmd = (self.cmd_ + " -key " + self.PrivateKey() +
               " -trusted_cert_file " + self.ca_.RootCertificate() +
               " -sqlite_db " + self.Database() +
               " -tree_signing_frequency_seconds 1" +
               " -logtostderr")
        logging.info("RUN: " + cmd)
        args = shlex.split(cmd)
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE)
        with timeout(10):
            while self.proc.stdout.readline() != "READY\n":
                continue

RootConfig = """[ req ]
distinguished_name=req_distinguished_name
prompt=no
x509_extensions=v3_ca

[ req_distinguished_name ]
countryName=GB
stateOrProvinceName=Wales
localityName=Erw Wen
0.organizationName=Certificate Transparency Test CA

[ v3_ca ]
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer:always
basicConstraints=CA:TRUE
"""

CAConfig = """[ ca ]
default_ca = CA_default

[ CA_default ]
default_startdate = 120601000000Z
default_enddate   = 220601000000Z
default_md	  = sha1
unique_subject	  = no
email_in_dn	  = no
policy	          = policy_default
serial            = {serial}
database          = {database}

[ policy_default ]
countryName	    = supplied
organizationName    = supplied
stateOrProvinceName = optional
localityName	    = optional
commonName          = optional
"""

RequestConfig = """[ req ]
distinguished_name=req_distinguished_name
prompt=no

[ req_distinguished_name ]
countryName=GB
stateOrProvinceName=Wales
localityName=Erw Wen
0.organizationName={subject}

# For the precert
[ pre ]
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer:always
basicConstraints=CA:FALSE
1.3.6.1.4.1.11129.2.4.3=critical,ASN1:NULL

# For the simple cert, without embedded proof extensions
[ simple ]
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer:always
basicConstraints=CA:FALSE

# For the cert with an embedded proof
[ embedded ]
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer:always
basicConstraints=CA:FALSE
"""

def WriteFile(name, content):
    with open(name, "w") as f:
        f.write(content)

class CA:
    def __init__(self, base):
        self.base_ = base

        os.mkdir(self.Directory())
        
        open(self.Database(), "w")
        WriteFile(self.Serial(), "0000000000000001")

        WriteFile(self.RootConfig(), RootConfig)
        ca_config = CAConfig.format(database = self.Database(),
                                    serial = self.Serial())
        WriteFile(self.CAConfig(), ca_config)

        self.GenerateRootCertificate()

        os.mkdir(self.IssuedCertificates())

    def Directory(self):
        """Where the CA does house-keeping"""
        return self.base_ + "-housekeeping"

    def Database(self):
        return self.Directory() + "/database"

    def Serial(self):
        return self.Directory() + "/serial"

    def RootConfig(self):
        return self.base_ + "-root-config"

    def CAConfig(self):
        return self.base_ + "-ca-config"

    def PrivateKey(self):
        return self.base_ + "-private-key.pem"

    def RootCertificate(self):
        return self.base_ + "-cert.pem"

    def TempFile(self, name):
        return self.base_ + "-temp-" + name

    def RequestConfig(self):
        return self.TempFile("req-config")

    def IssuedCertificates(self):
        return self.base_ + "-issued"

    def IssuedFile(self, name, subject):
        return self.IssuedCertificates() + "/" + name + "-" + subject + ".pem"

    def IssuedPrivateKey(self, subject):
        return self.IssuedFile("private-key", subject)

    def IssuedCertificate(self, subject):
        return self.IssuedFile("certificate", subject)

    def GenerateRootCertificate(self):
        csr = self.TempFile("csr")
        OpenSSL("req",
                "-new",
                "-newkey", "rsa:2048",
                "-keyout", self.PrivateKey(),
                "-out", csr,
                "-config", self.RootConfig(),
                "-nodes")
        OpenSSL("ca",
                "-in", csr,
                "-selfsign",
                "-keyfile", self.PrivateKey(),
                "-config", self.CAConfig(),
                "-extfile", self.RootConfig(),
                "-extensions", "v3_ca",
                "-outdir", self.Directory(),
                "-out", self.RootCertificate(),
                "-batch")

    def CreateAndLogCert(self, ct_server, subject):
        WriteFile(self.RequestConfig(), RequestConfig.format(subject=subject))
        csr = self.TempFile("csr")
        OpenSSL("req",
                "-new",
                "-newkey", "rsa:1024",
                "-keyout", self.IssuedPrivateKey(subject),
                "-out", csr,
                "-config", self.RequestConfig(),
                "-nodes")
        OpenSSL("ca",
                "-in", csr,
                "-cert", self.RootCertificate(),
                "-keyfile", self.PrivateKey(),
                "-config", self.CAConfig(),
                "-extfile", self.RequestConfig(),
                "-extensions", "simple",
                "-outdir", self.Directory(),
                "-out", self.IssuedCertificate(subject),
                "-batch")

        # Reverse the order of these to show a bug in ct-server where
        # it accepts the CA cert even though there's a redundant extra
        # cert in the chain. At least, I think its a bug.
        certs = (open(self.IssuedCertificate(subject)).read()
                 + open(self.RootCertificate()).read())
        chain_file = self.TempFile("chain")
        WriteFile(chain_file, certs)
        subprocess.check_call(("client/ct", "upload",
                               "-ct_server_submission", chain_file,
                               "-ct_server", ct_server.URL(),
                               "-ct_server_public_key", ct_server.PublicKey(),
                               "-ct_server_response_out", self.TempFile("sct")))

logging.basicConfig(level="WARNING")

# Set up our test CA
ca = CA(tmpdir + "/ct-test-ca")

# Run a CT server
ct_cmd = basepath + "/ct-server"
ct_server = CTServer(ct_cmd, tmpdir + "/ct-test", ca)
ct_server.Run()

# Add nn certs to the CT server
for x in range(NUMBER_OF_CERTS):
    ca.CreateAndLogCert(ct_server, "TestCertificate" + str(x))

# Make sure server has had enough time to assimilate all certs
time.sleep(2)

# We'll need the DB for the DNS server
db = ct_server.Database()
# Kill the CT server (shared database access not currently supported)
del ct_server

# Now run the DNS server from the same database
server_cmd = basepath + "/ct-dns-server --port=1111 --domain=example.com. --db=" + db
runner = DNSServerRunner()
runner.Run(server_cmd)

# Get the STH
lookup = CTDNSLookup(['127.0.0.1'], 1111)

sth = lookup.GetSTH()
logging.info("sth = " + str(sth))
logging.info("size = " + str(sth.tree_size))

assert sth.tree_size == NUMBER_OF_CERTS

# test all the entries
for index in range(NUMBER_OF_CERTS):
    leaf_hash = lookup.GetLeafHash(index)
    logging.info("index = " + str(index) + " hash = " + leaf_hash)

    verifier = merkle.MerkleVerifier()
    audit_path = []
    for level in range(0, verifier.audit_path_length(index, sth.tree_size)):
        hash = lookup.GetEntry(level, index, sth.tree_size)
        logging.info("hash = " + hash)
        audit_path.append(base64.b64decode(hash))

    logging.info("path = " + str(map(base64.b64encode, audit_path)))

    assert verifier.verify_leaf_hash_inclusion(base64.b64decode(leaf_hash),
                                               index, audit_path, sth)

print "DNS Server test passed"
