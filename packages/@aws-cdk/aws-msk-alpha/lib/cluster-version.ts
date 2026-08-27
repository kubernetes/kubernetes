/**
 * Available features for a given Kafka version
 */
export interface KafkaVersionFeatures {
  /**
   * Whether the Kafka version supports tiered storage mode.
   *
   * @see https://docs.aws.amazon.com/msk/latest/developerguide/msk-tiered-storage.html#msk-tiered-storage-requirements
   * @default false
   */
  readonly tieredStorage?: boolean;
}

/**
 * Kafka cluster version
 */
export class KafkaVersion {
  /**
   * **Deprecated by Amazon MSK. You can't create a Kafka cluster with a deprecated version.**
   *
   * Kafka version 1.1.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V1_1_1 = KafkaVersion.of('1.1.1');

  /**
   * **Deprecated by Amazon MSK. You can't create a Kafka cluster with a deprecated version.**
   *
   * Kafka version 2.1.0
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_1_0 = KafkaVersion.of('2.1.0');

  /**
   * Kafka version 2.2.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_2_1 = KafkaVersion.of('2.2.1');

  /**
   * Kafka version 2.3.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_3_1 = KafkaVersion.of('2.3.1');

  /**
   * **Deprecated by Amazon MSK. You can't create a Kafka cluster with a deprecated version.**
   *
   * Kafka version 2.4.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_4_1 = KafkaVersion.of('2.4.1');

  /**
   * Kafka version 2.4.1.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_4_1_1 = KafkaVersion.of('2.4.1.1');

  /**
   * Kafka version 2.5.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_5_1 = KafkaVersion.of('2.5.1');

  /**
   * Kafka version 2.6.0
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_6_0 = KafkaVersion.of('2.6.0');

  /**
   * Kafka version 2.6.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_6_1 = KafkaVersion.of('2.6.1');

  /**
   * Kafka version 2.6.2
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_6_2 = KafkaVersion.of('2.6.2');

  /**
   * Kafka version 2.6.3
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_6_3 = KafkaVersion.of('2.6.3');

  /**
   * Kafka version 2.7.0
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_7_0 = KafkaVersion.of('2.7.0');

  /**
   * Kafka version 2.7.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_7_1 = KafkaVersion.of('2.7.1');

  /**
   * Kafka version 2.7.2
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_7_2 = KafkaVersion.of('2.7.2');

  /**
   * Kafka version 2.8.0
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_8_0 = KafkaVersion.of('2.8.0');

  /**
   * Kafka version 2.8.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_8_1 = KafkaVersion.of('2.8.1');

  /**
   * AWS MSK Kafka version 2.8.2.tiered
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V2_8_2_TIERED = KafkaVersion.of('2.8.2.tiered', { tieredStorage: true });

  /**
   * Kafka version 3.1.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V3_1_1 = KafkaVersion.of('3.1.1');

  /**
   * Kafka version 3.2.0
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V3_2_0 = KafkaVersion.of('3.2.0');

  /**
   * Kafka version 3.3.1
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V3_3_1 = KafkaVersion.of('3.3.1');

  /**
   * Kafka version 3.3.2
   *
   * @deprecated use the latest runtime instead
   */
  public static readonly V3_3_2 = KafkaVersion.of('3.3.2');

  /**
   * Kafka version 3.4.0
   */
  public static readonly V3_4_0 = KafkaVersion.of('3.4.0');

  /**
   * Kafka version 3.5.1
   */
  public static readonly V3_5_1 = KafkaVersion.of('3.5.1');

  /**
   * Kafka version 3.6.0
   */
  public static readonly V3_6_0 = KafkaVersion.of('3.6.0', { tieredStorage: true });

  /**
   * Kafka version 3.7.x with ZooKeeper metadata mode support
   *
   * @see https://docs.aws.amazon.com/msk/latest/developerguide/metadata-management.html#msk-get-connection-string
   */
  public static readonly V3_7_X = KafkaVersion.of('3.7.x', { tieredStorage: true });

  /**
   * Kafka version 3.7.x with KRaft (Apache Kafka Raft) metadata mode support
   *
   * @see https://docs.aws.amazon.com/msk/latest/developerguide/metadata-management.html#kraft-intro
   */
  public static readonly V3_7_X_KRAFT = KafkaVersion.of('3.7.x.kraft', { tieredStorage: true });

  /**
   * Kafka version 3.8.x with ZooKeeper metadata mode support
   *
   * @see https://docs.aws.amazon.com/msk/latest/developerguide/metadata-management.html#msk-get-connection-string
   */
  public static readonly V3_8_X = KafkaVersion.of('3.8.x', { tieredStorage: true });

  /**
   * Kafka version 3.8.x with KRaft (Apache Kafka Raft) metadata mode support
   *
   * @see https://docs.aws.amazon.com/msk/latest/developerguide/metadata-management.html#kraft-intro
   */
  public static readonly V3_8_X_KRAFT = KafkaVersion.of('3.8.x.kraft', { tieredStorage: true });

  /**
   * Kafka version 3.9.x with ZooKeeper metadata mode support
   *
   * @see https://docs.aws.amazon.com/msk/latest/developerguide/metadata-management.html#msk-get-connection-string
   */
  public static readonly V3_9_X = KafkaVersion.of('3.9.x', { tieredStorage: true });

  /**
   * Kafka version 3.9.x with KRaft (Apache Kafka Raft) metadata mode support
   *
   * @see https://docs.aws.amazon.com/msk/latest/developerguide/metadata-management.html#kraft-intro
   */
  public static readonly V3_9_X_KRAFT = KafkaVersion.of('3.9.x.kraft', { tieredStorage: true });

  /**
   * Kafka version 4.0.x with KRaft (Apache Kafka Raft) metadata mode support
   *
   * @see https://docs.aws.amazon.com/msk/latest/developerguide/metadata-management.html#kraft-intro
   */
  public static readonly V4_0_X_KRAFT = KafkaVersion.of('4.0.x.kraft', { tieredStorage: true });

  /**
   * Kafka version 4.1.x with KRaft (Apache Kafka Raft) metadata mode support
   *
   * @see https://docs.aws.amazon.com/msk/latest/developerguide/metadata-management.html#kraft-intro
   */
  public static readonly V4_1_X_KRAFT = KafkaVersion.of('4.1.x.kraft', { tieredStorage: true });

  /**
   * Kafka version 4.2.x with KRaft (Apache Kafka Raft) metadata mode support
   *
   * Currently only supported on MSK Express brokers.
   *
   * @see https://docs.aws.amazon.com/msk/latest/developerguide/metadata-management.html#kraft-intro
   */
  public static readonly V4_2_X_KRAFT = KafkaVersion.of('4.2.x.kraft', { tieredStorage: true });

  /**
   * Custom cluster version
   * @param version custom version number
   */
  public static of(version: string, features?: KafkaVersionFeatures) {
    return new KafkaVersion(version, features);
  }

  /**
   *
   * @param version cluster version number
   * @param features features for the cluster version
   */
  private constructor(public readonly version: string, public readonly features?: KafkaVersionFeatures) {}

  /**
   * Checks if the cluster version supports tiered storage mode.
   */
  public isTieredStorageCompatible(): boolean {
    return this.features?.tieredStorage ?? false;
  }
}
