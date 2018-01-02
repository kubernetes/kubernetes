package org.certificatetransparency.ctlog;

import org.certificatetransparency.ctlog.serialization.CryptoDataLoader;

import java.io.File;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.util.Arrays;

/**
 * Holds information about the log: Mainly, its public key and log ID (which is calculated
 * from the Log ID).
 * Ideally created from a file with the Log's public key in PEM encoding.
 */
public class LogInfo {
  private final PublicKey logKey;
  private final byte[] logId;

  /**
   * C'tor.
   *
   * @param logKey Public key of the log.
   */
  public LogInfo(PublicKey logKey) {
    this.logKey = logKey;
    logId = calculateLogId(logKey);
  }

  byte[] getID() {
    return logId;
  }

  public PublicKey getKey() {
    return logKey;
  }

  public String getSignatureAlgorithm() {
    return logKey.getAlgorithm();
  }

  public boolean isSameLogId(byte[] idToCheck) {
    return Arrays.equals(getID(), idToCheck);
  }

  private static byte[] calculateLogId(PublicKey logKey) {
    try {
      MessageDigest sha256 = MessageDigest.getInstance("SHA-256");
      sha256.update(logKey.getEncoded());
      return sha256.digest();

    } catch (NoSuchAlgorithmException e) {
      throw new UnsupportedCryptoPrimitiveException("Missing SHA-256", e);
    }
  }

  /**
   * Creates a LogInfo instance from the Log's public key file.
   *
   * @param pemKeyFilePath Path of the log's public key file.
   * @return new LogInfo instance.
   */
  public static LogInfo fromKeyFile(String pemKeyFilePath) {
    PublicKey logPublicKey = CryptoDataLoader.keyFromFile(new File(pemKeyFilePath));
    return new LogInfo(logPublicKey);
  }
}
