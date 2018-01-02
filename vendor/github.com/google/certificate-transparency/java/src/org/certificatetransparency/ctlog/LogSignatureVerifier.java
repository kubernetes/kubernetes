package org.certificatetransparency.ctlog;

import static org.certificatetransparency.ctlog.serialization.CTConstants.LOG_ENTRY_TYPE_LENGTH;
import static org.certificatetransparency.ctlog.serialization.CTConstants.MAX_CERTIFICATE_LENGTH;
import static org.certificatetransparency.ctlog.serialization.CTConstants.MAX_EXTENSIONS_LENGTH;
import static org.certificatetransparency.ctlog.serialization.CTConstants.TIMESTAMP_LENGTH;
import static org.certificatetransparency.ctlog.serialization.CTConstants.VERSION_LENGTH;

import com.google.common.base.Preconditions;

import org.bouncycastle.asn1.ASN1InputStream;
import org.bouncycastle.asn1.ASN1ObjectIdentifier;
import org.bouncycastle.asn1.x500.X500Name;
import org.bouncycastle.asn1.x509.Extension;
import org.bouncycastle.asn1.x509.Extensions;
import org.bouncycastle.asn1.x509.TBSCertificate;
import org.bouncycastle.asn1.x509.V3TBSCertificateGenerator;
import org.certificatetransparency.ctlog.proto.Ct;
import org.certificatetransparency.ctlog.serialization.CTConstants;
import org.certificatetransparency.ctlog.serialization.Serializer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.Signature;
import java.security.SignatureException;
import java.security.cert.Certificate;
import java.security.cert.CertificateEncodingException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;


/**
 * Verifies signatures from a given CT Log.
 */
public class LogSignatureVerifier {
  public static final String X509_AUTHORITY_KEY_IDENTIFIER = "2.5.29.35";
  private final LogInfo logInfo;

  /**
   * Creates a new LogSignatureVerifier which is associated with a single log.
   * @param logInfo information of the log this verifier is to be associated with.
   */
  public LogSignatureVerifier(LogInfo logInfo) {
    this.logInfo = logInfo;
  }

  private static class IssuerInformation {
    private final X500Name name;
    private final byte[] keyHash;
    private final Extension x509authorityKeyIdentifier;
    private final boolean issuedByPreCertificateSigningCert;

    IssuerInformation(
        X500Name name, byte[] keyHash, Extension x509authorityKeyIdentifier,
        boolean issuedByPreCertificateSigningCert) {
      this.name = name;
      this.keyHash = keyHash;
      this.x509authorityKeyIdentifier = x509authorityKeyIdentifier;
      this.issuedByPreCertificateSigningCert = issuedByPreCertificateSigningCert;
    }

    X500Name getName() {
      return name;
    }

    byte[] getKeyHash() {
      return keyHash;
    }

    Extension getX509authorityKeyIdentifier() {
      return x509authorityKeyIdentifier;
    }

    boolean issuedByPreCertificateSigningCert() {
      return issuedByPreCertificateSigningCert;
    }
  }

  static IssuerInformation issuerInformationFromPreCertificateSigningCert(
      Certificate certificate, byte[] keyHash) {
    try (ASN1InputStream aIssuerIn = new ASN1InputStream(certificate.getEncoded())) {
      org.bouncycastle.asn1.x509.Certificate parsedIssuerCert =
          org.bouncycastle.asn1.x509.Certificate.getInstance(aIssuerIn.readObject());

      Extensions issuerExtensions = parsedIssuerCert.getTBSCertificate().getExtensions();
      Extension x509authorityKeyIdentifier = null;
      if (issuerExtensions != null) {
        x509authorityKeyIdentifier =
            issuerExtensions.getExtension(new ASN1ObjectIdentifier(X509_AUTHORITY_KEY_IDENTIFIER));
      }

      return new IssuerInformation(
          parsedIssuerCert.getIssuer(), keyHash, x509authorityKeyIdentifier, true);
    } catch (CertificateEncodingException e) {
      throw new CertificateTransparencyException(
          "Certificate could not be encoded: " + e.getMessage(), e);
    } catch (IOException e) {
      throw new CertificateTransparencyException(
          "Error during ASN.1 parsing of certificate: " + e.getMessage(), e);
    }
  }

  // Produces issuer information in case the PreCertificate is signed by a regular CA cert,
  // not PreCertificate Signing Cert. In this case, the only thing that's needed is the
  // issuer key hash - the Precertificate will already have the right value for the issuer
  // name and K509 Authority Key Identifier extension.
  static IssuerInformation issuerInformationFromCertificateIssuer(Certificate certificate) {
    return new IssuerInformation(null, getKeyHash(certificate), null, false);
  }

  /**
   * Verifies the CT Log's signature over the SCT and certificate.
   * Works for the following cases:
   * * Ordinary X509 certificate sent to the log.
   * * PreCertificate signed by an ordinary CA certificate.
   * * PreCertificate signed by a PreCertificate Signing Cert. In this case the PreCertificate
   *   signing certificate must be 2nd on the chain, the CA cert itself 3rd.
   *
   * It does not work for verifying a final certificate with the CT extension.
   * TODO(eranm): Add the ability to remove the CT extension and verify a final certificate.
   *
   * @param sct SignedCertificateTimestamp received from the log.
   * @param chain The certificates chain as sent to the log.
   * @return true if the log's signature over this SCT can be verified, false otherwise.
   */
  public boolean verifySignature(Ct.SignedCertificateTimestamp sct, List<Certificate> chain) {
    if (!logInfo.isSameLogId(sct.getId().getKeyId().toByteArray())) {
      throw new CertificateTransparencyException(String.format(
          "Log ID of SCT (%s) does not match this log's ID.", sct.getId().getKeyId()));
    }

    X509Certificate leafCert = (X509Certificate) chain.get(0);
    if (!CertificateInfo.isPreCertificate(leafCert)) {
      byte[] toVerify = serializeSignedSCTData(leafCert, sct);
      return verifySCTSignatureOverBytes(sct, toVerify);
    } else {
      Preconditions.checkArgument(chain.size() >= 2,
          "Chain with PreCertificate must contain issuer.");
      // PreCertificate
      Certificate issuerCert = chain.get(1);
      IssuerInformation issuerInformation;
      if (!CertificateInfo.isPreCertificateSigningCert(issuerCert)) {
        issuerInformation = issuerInformationFromCertificateIssuer(issuerCert);
      } else {
        Preconditions.checkArgument(chain.size() >= 3,
            "Chain with PreCertificate signed by PreCertificate Signing Cert must contain issuer.");
        issuerInformation = issuerInformationFromPreCertificateSigningCert(
            issuerCert, getKeyHash(chain.get(2)));
      }
      return verifySCTOverPreCertificate(sct, leafCert, issuerInformation);
    }
  }

  /**
   * Verifies the CT Log's signature over the SCT and leaf certificate.
   * @param sct SignedCertificateTimestamp received from the log.
   * @param leafCert leaf certificate sent to the log.
   * @return true if the log's signature over this SCT can be verified, false otherwise.
   */
  boolean verifySignature(Ct.SignedCertificateTimestamp sct, Certificate leafCert) {
    if (!logInfo.isSameLogId(sct.getId().getKeyId().toByteArray())) {
      throw new CertificateTransparencyException(String.format(
          "Log ID of SCT (%s) does not match this log's ID.", sct.getId().getKeyId()));
    }
    byte[] toVerify = serializeSignedSCTData(leafCert, sct);

    return verifySCTSignatureOverBytes(sct, toVerify);
  }

  /**
   * Verifies the CT Log's signature over the SCT and PreCertificate.
   *
   * @param sct SignedCertificateTimestamp received from the log.
   * @param preCertificate PreCertificate sent to the log for addition.
   * @param issuerInfo Information on the issuer which will ultimately sign this PreCertificate.
   *                   If the PreCertificate was signed using by a PreCertificate Signing Cert,
   *                   the issuerInfo contains data on the final CA certificate used for signing.
   * @return true if the SCT verifies, false otherwise.
   */
  boolean verifySCTOverPreCertificate(Ct.SignedCertificateTimestamp sct,
                                      X509Certificate preCertificate,
                                      IssuerInformation issuerInfo) {
    Preconditions.checkNotNull(issuerInfo, "At the very least, the issuer key hash is needed.");
    //TODO(eranm): Remove this restriction after this method knows how to strip the CT
    // extension from a final certificate.
    Preconditions.checkArgument(
        CertificateInfo.isPreCertificate(preCertificate),
        "PreCertificate must contain the poison extension");

    TBSCertificate preCertificateTBS = createTbsForVerification(preCertificate, issuerInfo);
    try {
      byte[] toVerify = serializeSignedSCTDataForPreCertificate(
          preCertificateTBS.getEncoded(), issuerInfo.getKeyHash(), sct);
      return verifySCTSignatureOverBytes(sct, toVerify);
    } catch (IOException e) {
      throw new CertificateTransparencyException(
          "TBSCertificate part could not be encoded: " + e.getMessage(), e);
    }
  }

  private TBSCertificate createTbsForVerification(
      X509Certificate preCertificate, IssuerInformation issuerInformation) {
    Preconditions.checkArgument(preCertificate.getVersion() >= 3);
    // We have to use bouncycastle's certificate parsing code because Java's X509 certificate
    // parsing discards the order of the extensions. The signature from SCT we're verifying
    // is over the TBSCertificate in its original form, including the order of the extensions.
    // Get the list of extensions, in its original order, minus the poison extension.
    try (ASN1InputStream aIn = new ASN1InputStream(preCertificate.getEncoded())) {
      org.bouncycastle.asn1.x509.Certificate parsedPreCertificate =
          org.bouncycastle.asn1.x509.Certificate.getInstance(aIn.readObject());
      // Make sure that we have the X509akid of the real issuer if:
      // The PreCertificate has this extension, AND:
      // The PreCertificate was signed by a PreCertificate signing cert.
      if (hasX509AuthorityKeyIdentifier(parsedPreCertificate) &&
          issuerInformation.issuedByPreCertificateSigningCert()) {
        Preconditions.checkArgument(issuerInformation.getX509authorityKeyIdentifier() != null);
      }

      List<Extension> orderedExtensions = getExtensionsWithoutPoison(
          parsedPreCertificate.getTBSCertificate().getExtensions(),
          issuerInformation.getX509authorityKeyIdentifier());

      V3TBSCertificateGenerator tbsCertificateGenerator = new V3TBSCertificateGenerator();
      TBSCertificate tbsPart = parsedPreCertificate.getTBSCertificate();
      // Copy certificate.
      // Version 3 is implied by the generator.
      tbsCertificateGenerator.setSerialNumber(tbsPart.getSerialNumber());
      tbsCertificateGenerator.setSignature(tbsPart.getSignature());
      if (issuerInformation.getName() != null) {
        tbsCertificateGenerator.setIssuer(issuerInformation.getName());
      } else {
        tbsCertificateGenerator.setIssuer(tbsPart.getIssuer());
      }
      tbsCertificateGenerator.setStartDate(tbsPart.getStartDate());
      tbsCertificateGenerator.setEndDate(tbsPart.getEndDate());
      tbsCertificateGenerator.setSubject(tbsPart.getSubject());
      tbsCertificateGenerator.setSubjectPublicKeyInfo(tbsPart.getSubjectPublicKeyInfo());
      tbsCertificateGenerator.setIssuerUniqueID(tbsPart.getIssuerUniqueId());
      tbsCertificateGenerator.setSubjectUniqueID(tbsPart.getSubjectUniqueId());
      tbsCertificateGenerator.setExtensions(
          new Extensions(orderedExtensions.toArray(new Extension[]{})));
      return tbsCertificateGenerator.generateTBSCertificate();
    } catch (CertificateException e) {
      throw new CertificateTransparencyException("Certificate error: " + e.getMessage(), e);
    } catch (IOException e) {
      throw new CertificateTransparencyException("Error deleting extension: " + e.getMessage(), e);
    }
  }

  private static boolean hasX509AuthorityKeyIdentifier(
      org.bouncycastle.asn1.x509.Certificate cert) {
    Extensions extensions = cert.getTBSCertificate().getExtensions();
    return extensions.getExtension(new ASN1ObjectIdentifier(X509_AUTHORITY_KEY_IDENTIFIER)) != null;
  }

  private List<Extension> getExtensionsWithoutPoison(
      Extensions extensions,
      Extension replacementX509authorityKeyIdentifier) {
    ASN1ObjectIdentifier[] extensionsOidsArray = extensions.getExtensionOIDs();
    Iterator<ASN1ObjectIdentifier> extensionsOids = Arrays.asList(extensionsOidsArray).iterator();

    // Order is important, which is why a list is used.
    ArrayList<Extension> outputExtensions = new ArrayList<Extension>();
    while (extensionsOids.hasNext()) {
      ASN1ObjectIdentifier extn = extensionsOids.next();
      String extnId = extn.getId();
      if (extnId.equals(CTConstants.POISON_EXTENSION_OID)) {
        // Do nothing - skip copying this extension
      } else if ((extnId.equals(X509_AUTHORITY_KEY_IDENTIFIER)) &&
          (replacementX509authorityKeyIdentifier != null)) {
        // Use the real issuer's authority key identifier, since it's present.
        outputExtensions.add(replacementX509authorityKeyIdentifier);
      } else {
        // Copy the extension as-is.
        outputExtensions.add(extensions.getExtension(extn));
      }
    }
    return outputExtensions;
  }

  private boolean verifySCTSignatureOverBytes(Ct.SignedCertificateTimestamp sct, byte[] toVerify) {
    if (!logInfo.getSignatureAlgorithm().equals("EC")) {
      throw new CertificateTransparencyException(
          String.format("Non-EC signature %s not supported yet",
              logInfo.getSignatureAlgorithm()));
    }

    try {
      Signature signature = Signature.getInstance("SHA256withECDSA");
      signature.initVerify(logInfo.getKey());
      signature.update(toVerify);
      return signature.verify(sct.getSignature().getSignature().toByteArray());
    } catch (SignatureException e) {
      throw new CertificateTransparencyException("Signature object not properly initialized or"
          + " signature from SCT is improperly encoded.", e);
    } catch (InvalidKeyException e) {
      throw new CertificateTransparencyException("Log's public key cannot be used", e);
    } catch (NoSuchAlgorithmException e) {
      throw new UnsupportedCryptoPrimitiveException(
          "Sha-256 with ECDSA not supported by this JVM", e);
    }
  }

  static byte[] serializeSignedSCTData(Certificate certificate,
                                       Ct.SignedCertificateTimestamp sct) {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    serializeCommonSCTFields(sct, bos);
    Serializer.writeUint(bos, Ct.LogEntryType.X509_ENTRY_VALUE, LOG_ENTRY_TYPE_LENGTH);
    try {
      Serializer.writeVariableLength(bos, certificate.getEncoded(), MAX_CERTIFICATE_LENGTH);
    } catch (CertificateEncodingException e) {
      throw new CertificateTransparencyException("Error encoding certificate", e);
    }
    Serializer.writeVariableLength(bos, sct.getExtensions().toByteArray(),
        MAX_EXTENSIONS_LENGTH);

    return bos.toByteArray();
  }

  static byte[] serializeSignedSCTDataForPreCertificate(byte[] preCertBytes,
                                                        byte[] issuerKeyHash,
                                                        Ct.SignedCertificateTimestamp sct) {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    serializeCommonSCTFields(sct, bos);
    Serializer.writeUint(bos, Ct.LogEntryType.PRECERT_ENTRY_VALUE, LOG_ENTRY_TYPE_LENGTH);
    Serializer.writeFixedBytes(bos, issuerKeyHash);
    Serializer.writeVariableLength(bos, preCertBytes, MAX_CERTIFICATE_LENGTH);
    Serializer.writeVariableLength(bos, sct.getExtensions().toByteArray(), MAX_EXTENSIONS_LENGTH);
    return bos.toByteArray();
  }

  private static byte[] getKeyHash(Certificate signerCert) {
    try {
      MessageDigest sha256 = MessageDigest.getInstance("SHA-256");
      return sha256.digest(signerCert.getPublicKey().getEncoded());
    } catch (NoSuchAlgorithmException e) {
      throw new UnsupportedCryptoPrimitiveException("SHA-256 not supported: " + e.getMessage(), e);
    }
  }

  private static void serializeCommonSCTFields(
      Ct.SignedCertificateTimestamp sct, ByteArrayOutputStream bos) {
    Preconditions.checkArgument(sct.getVersion().equals(Ct.Version.V1),
        "Can only serialize SCT v1 for now.");
    Serializer.writeUint(bos, sct.getVersion().getNumber(), VERSION_LENGTH); // ct::V1
    Serializer.writeUint(bos, 0, 1); // ct::CERTIFICATE_TIMESTAMP
    Serializer.writeUint(bos, sct.getTimestamp(), TIMESTAMP_LENGTH); // Timestamp
  }
}
