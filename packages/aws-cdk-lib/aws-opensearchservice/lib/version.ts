/**
 * OpenSearch version
 */
export class EngineVersion {
  /** AWS Elasticsearch 1.5 */
  public static readonly ELASTICSEARCH_1_5 = EngineVersion.elasticsearch('1.5');

  /** AWS Elasticsearch 2.3 */
  public static readonly ELASTICSEARCH_2_3 = EngineVersion.elasticsearch('2.3');

  /** AWS Elasticsearch 5.1 */
  public static readonly ELASTICSEARCH_5_1 = EngineVersion.elasticsearch('5.1');

  /** AWS Elasticsearch 5.3 */
  public static readonly ELASTICSEARCH_5_3 = EngineVersion.elasticsearch('5.3');

  /** AWS Elasticsearch 5.5 */
  public static readonly ELASTICSEARCH_5_5 = EngineVersion.elasticsearch('5.5');

  /** AWS Elasticsearch 5.6 */
  public static readonly ELASTICSEARCH_5_6 = EngineVersion.elasticsearch('5.6');

  /** AWS Elasticsearch 6.0 */
  public static readonly ELASTICSEARCH_6_0 = EngineVersion.elasticsearch('6.0');

  /** AWS Elasticsearch 6.2 */
  public static readonly ELASTICSEARCH_6_2 = EngineVersion.elasticsearch('6.2');

  /** AWS Elasticsearch 6.3 */
  public static readonly ELASTICSEARCH_6_3 = EngineVersion.elasticsearch('6.3');

  /** AWS Elasticsearch 6.4 */
  public static readonly ELASTICSEARCH_6_4 = EngineVersion.elasticsearch('6.4');

  /** AWS Elasticsearch 6.5 */
  public static readonly ELASTICSEARCH_6_5 = EngineVersion.elasticsearch('6.5');

  /** AWS Elasticsearch 6.7 */
  public static readonly ELASTICSEARCH_6_7 = EngineVersion.elasticsearch('6.7');

  /** AWS Elasticsearch 6.8 */
  public static readonly ELASTICSEARCH_6_8 = EngineVersion.elasticsearch('6.8');

  /** AWS Elasticsearch 7.1 */
  public static readonly ELASTICSEARCH_7_1 = EngineVersion.elasticsearch('7.1');

  /** AWS Elasticsearch 7.4 */
  public static readonly ELASTICSEARCH_7_4 = EngineVersion.elasticsearch('7.4');

  /** AWS Elasticsearch 7.7 */
  public static readonly ELASTICSEARCH_7_7 = EngineVersion.elasticsearch('7.7');

  /** AWS Elasticsearch 7.8 */
  public static readonly ELASTICSEARCH_7_8 = EngineVersion.elasticsearch('7.8');

  /** AWS Elasticsearch 7.9 */
  public static readonly ELASTICSEARCH_7_9 = EngineVersion.elasticsearch('7.9');

  /** AWS Elasticsearch 7.10 */
  public static readonly ELASTICSEARCH_7_10 = EngineVersion.elasticsearch('7.10');

  /** AWS OpenSearch 1.0 */
  public static readonly OPENSEARCH_1_0 = EngineVersion.openSearch('1.0');

  /** AWS OpenSearch 1.1 */
  public static readonly OPENSEARCH_1_1 = EngineVersion.openSearch('1.1');

  /** AWS OpenSearch 1.2 */
  public static readonly OPENSEARCH_1_2 = EngineVersion.openSearch('1.2');

  /** AWS OpenSearch 1.3 */
  public static readonly OPENSEARCH_1_3 = EngineVersion.openSearch('1.3');

  /**
   * AWS OpenSearch 2.3
   *
   * OpenSearch 2.3 is now available on Amazon OpenSearch Service across 26
   * regions globally. Please refer to the AWS Region Table for more
   * information about Amazon OpenSearch Service availability:
   * https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/
   * */
  public static readonly OPENSEARCH_2_3 = EngineVersion.openSearch('2.3');

  /** AWS OpenSearch 2.5 */
  public static readonly OPENSEARCH_2_5 = EngineVersion.openSearch('2.5');

  /** AWS OpenSearch 2.7 */
  public static readonly OPENSEARCH_2_7 = EngineVersion.openSearch('2.7');

  /** AWS OpenSearch 2.9 */
  public static readonly OPENSEARCH_2_9 = EngineVersion.openSearch('2.9');

  /**
   * AWS OpenSearch 2.10
   * @deprecated use latest version of the OpenSearch engine
   **/
  public static readonly OPENSEARCH_2_10 = EngineVersion.openSearch('2.10');

  /** AWS OpenSearch 2.11 */
  public static readonly OPENSEARCH_2_11 = EngineVersion.openSearch('2.11');

  /** AWS OpenSearch 2.13 */
  public static readonly OPENSEARCH_2_13 = EngineVersion.openSearch('2.13');

  /** AWS OpenSearch 2.15 */
  public static readonly OPENSEARCH_2_15 = EngineVersion.openSearch('2.15');

  /** AWS OpenSearch 2.17 */
  public static readonly OPENSEARCH_2_17 = EngineVersion.openSearch('2.17');

  /** AWS OpenSearch 2.19 */
  public static readonly OPENSEARCH_2_19 = EngineVersion.openSearch('2.19');

  /** AWS OpenSearch 3.1 */
  public static readonly OPENSEARCH_3_1 = EngineVersion.openSearch('3.1');

  /** AWS OpenSearch 3.3 */
  public static readonly OPENSEARCH_3_3 = EngineVersion.openSearch('3.3');

  /** AWS OpenSearch 3.5 */
  public static readonly OPENSEARCH_3_5 = EngineVersion.openSearch('3.5');

  /** AWS OpenSearch 3.7 */
  public static readonly OPENSEARCH_3_7 = EngineVersion.openSearch('3.7');

  /**
   * Custom ElasticSearch version
   * @param version custom version number
   */
  public static elasticsearch(version: string) { return new EngineVersion(`Elasticsearch_${version}`); }

  /**
   * Custom OpenSearch version
   * @param version custom version number
   */
  public static openSearch(version: string) { return new EngineVersion(`OpenSearch_${version}`); }

  /**
   * @param version engine version identifier
   */
  private constructor(public readonly version: string) { }
}
