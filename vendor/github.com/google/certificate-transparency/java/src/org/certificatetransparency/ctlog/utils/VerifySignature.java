package org.certificatetransparency.ctlog.utils;

import com.google.common.io.Files;
import com.google.protobuf.InvalidProtocolBufferException;
import org.certificatetransparency.ctlog.LogInfo;
import org.certificatetransparency.ctlog.LogSignatureVerifier;
import org.certificatetransparency.ctlog.proto.Ct;
import org.certificatetransparency.ctlog.serialization.CryptoDataLoader;
import org.certificatetransparency.ctlog.serialization.Deserializer;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.security.cert.Certificate;
import java.util.List;

/**
 * Utility for verifying a log's signature from an SCT.
 */
public class VerifySignature {
  public static void main(String[] args) throws IOException {
    if (args.length < 3) {
      System.out.println(String.format("Usage: %s <certificates chain> <sct> <log public key>",
          VerifySignature.class.getSimpleName()));
      return;
    }

    String pemFile = args[0];
    String sctFile = args[1];
    String logPublicKeyFile = args[2];

    List<Certificate> certs = CryptoDataLoader.certificatesFromFile(new File(pemFile));
    byte[] sctBytes = Files.toByteArray(new File(sctFile));

    Ct.SignedCertificateTimestamp sct;
    try {
      sct = Ct.SignedCertificateTimestamp.parseFrom(sctBytes);
    } catch (InvalidProtocolBufferException e) {
      System.out.println("Not a protocol buffer. Trying reading as binary");
      sct = Deserializer.parseSCTFromBinary(new ByteArrayInputStream(sctBytes));
    }

    System.out.println("Canned SCT: " + sct.toString());

    LogInfo logInfo = LogInfo.fromKeyFile(logPublicKeyFile);
    LogSignatureVerifier verifier = new LogSignatureVerifier(logInfo);
    if (verifier.verifySignature(sct, certs)) {
      System.out.println("Signature verified OK.");
    } else {
      System.out.println("Signature verification FAILURE.");
      System.exit(-1);
    }
  }
}
