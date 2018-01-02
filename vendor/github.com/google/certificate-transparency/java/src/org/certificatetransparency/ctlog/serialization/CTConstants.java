package org.certificatetransparency.ctlog.serialization;

/**
 * Constants used for serializing and de-serializing.
 */
public class CTConstants {
  // All in bytes.
  public static final int MAX_EXTENSIONS_LENGTH = (1 << 16) - 1;
  public static final int MAX_SIGNATURE_LENGTH = (1 << 16) - 1;
  public static final int KEY_ID_LENGTH = 32;
  public static final int TIMESTAMP_LENGTH = 8;
  public static final int VERSION_LENGTH = 1;
  public static final int LOG_ENTRY_TYPE_LENGTH = 2;
  public static final int HASH_ALG_LENGTH = 1;
  public static final int SIGNATURE_ALG_LENGTH = 1;
  public static final int MAX_CERTIFICATE_LENGTH = (1 << 24) - 1;

  // Useful OIDs
  public static final String PRECERTIFICATE_SIGNING_OID = "1.3.6.1.4.1.11129.2.4.4";
  public static final String POISON_EXTENSION_OID = "1.3.6.1.4.1.11129.2.4.3";
}
