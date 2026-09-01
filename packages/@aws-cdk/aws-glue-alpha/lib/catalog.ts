import type { CatalogReference, ICatalogRef } from 'aws-cdk-lib/aws-glue';
import { CfnCatalog, CfnDataCatalogEncryptionSettings } from 'aws-cdk-lib/aws-glue';
import type * as iam from 'aws-cdk-lib/aws-iam';
import type * as kms from 'aws-cdk-lib/aws-kms';
import { KeyGrants } from 'aws-cdk-lib/aws-kms';
import type { IResource } from 'aws-cdk-lib/core';
import { ArnFormat, Resource, Stack, Token, ValidationError } from 'aws-cdk-lib/core';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';

/**
 * The encryption-at-rest mode for a Glue Data Catalog.
 *
 * @see https://docs.aws.amazon.com/glue/latest/webapi/API_EncryptionAtRest.html#Glue-Type-EncryptionAtRest-CatalogEncryptionMode
 */
export enum CatalogEncryptionMode {
  /**
   * Encryption at rest is disabled.
   */
  DISABLED = 'DISABLED',

  /**
   * Server-side encryption (SSE) with an AWS KMS key.
   */
  SSE_KMS = 'SSE-KMS',

  /**
   * Server-side encryption (SSE) with an AWS KMS key, using a service role that
   * AWS Glue assumes to access the key on your behalf.
   */
  SSE_KMS_WITH_SERVICE_ROLE = 'SSE-KMS-WITH-SERVICE-ROLE',
}

/**
 * Encryption-at-rest configuration for a Glue Data Catalog.
 *
 * The Data Catalog encryption at rest and the connection password encryption
 * are independent: enabling one does not require the other, and each may use a
 * different KMS key.
 *
 * @see https://docs.aws.amazon.com/glue/latest/webapi/API_EncryptionAtRest.html
 */
export class DataCatalogEncryptionAtRest {
  /**
   * Disable encryption at rest for the Data Catalog.
   */
  public static disabled(): DataCatalogEncryptionAtRest {
    return new DataCatalogEncryptionAtRest(CatalogEncryptionMode.DISABLED);
  }

  /**
   * Encrypt the Data Catalog at rest with an AWS KMS key.
   *
   * @param key the KMS key to use. If omitted, an AWS-managed key is used and
   * the key is not exposed as a grantable resource.
   */
  public static kms(key?: kms.IKeyRef): DataCatalogEncryptionAtRest {
    return new DataCatalogEncryptionAtRest(CatalogEncryptionMode.SSE_KMS, key);
  }

  /**
   * Encrypt the Data Catalog at rest with an AWS KMS key, accessed through a
   * service role that AWS Glue assumes on your behalf.
   *
   * When a customer-managed `key` is provided, the `role` is automatically
   * granted `kms:Encrypt`/`kms:Decrypt`/`kms:GenerateDataKey*` on it.
   *
   * @param role the service role that AWS Glue assumes to access the key.
   * @param key the KMS key to use. If omitted, an AWS-managed key is used and
   * the key is not exposed as a grantable resource.
   */
  public static kmsWithServiceRole(role: iam.IRole, key?: kms.IKeyRef): DataCatalogEncryptionAtRest {
    return new DataCatalogEncryptionAtRest(CatalogEncryptionMode.SSE_KMS_WITH_SERVICE_ROLE, key, role);
  }

  /**
   * The encryption mode.
   */
  public readonly mode: CatalogEncryptionMode;

  /**
   * The customer-managed KMS key used for encryption at rest, if any.
   */
  public readonly kmsKey?: kms.IKeyRef;

  /**
   * The service role that AWS Glue assumes to access the KMS key, if any.
   */
  public readonly serviceRole?: iam.IRole;

  private constructor(mode: CatalogEncryptionMode, kmsKey?: kms.IKeyRef, serviceRole?: iam.IRole) {
    this.mode = mode;
    this.kmsKey = kmsKey;
    this.serviceRole = serviceRole;
  }
}

/**
 * Connection-password encryption configuration for a Glue Data Catalog.
 *
 * When enabled, the Data Catalog encrypts the password as part of
 * `CreateConnection` or `UpdateConnection` and stores it in the
 * `ENCRYPTED_PASSWORD` field of the connection properties. This is independent
 * from catalog encryption at rest, and may use a different KMS key.
 *
 * @see https://docs.aws.amazon.com/glue/latest/webapi/API_ConnectionPasswordEncryption.html
 */
export interface ConnectionPasswordEncryption {
  /**
   * The KMS key used to encrypt connection passwords.
   *
   * @default - an AWS-managed key is used and the key is not exposed as a grantable resource.
   */
  readonly kmsKey?: kms.IKeyRef;

  /**
   * Whether passwords remain encrypted in the responses of `GetConnection` and
   * `GetConnections`. This takes effect independently from catalog encryption.
   *
   * @default true
   */
  readonly returnConnectionPasswordEncrypted?: boolean;
}

/**
 * Encryption configuration for a Glue Data Catalog.
 *
 * Encryption is fixed at construction: a catalog either carries encryption
 * settings or it does not, which keeps its configuration easy to reason about
 * and avoids order-dependent mutation after the catalog is created.
 */
export interface CatalogEncryptionOptions {
  /**
   * Encryption-at-rest configuration for the catalog.
   *
   * @default - encryption at rest is not managed by CDK (the catalog default applies)
   */
  readonly encryptionAtRest?: DataCatalogEncryptionAtRest;

  /**
   * Connection-password encryption configuration for the catalog.
   *
   * @default - connection-password encryption is not managed by CDK
   */
  readonly connectionPasswordEncryption?: ConnectionPasswordEncryption;
}

/**
 * A Glue Data Catalog, either the implicit account-wide catalog or one created
 * as an `AWS::Glue::Catalog` resource.
 */
export interface ICatalog extends IResource, ICatalogRef {
  /**
   * The id of the catalog (for the account-wide catalog, the AWS account id).
   * @attribute
   */
  readonly catalogId: string;

  /**
   * The ARN of the catalog.
   * @attribute
   */
  readonly catalogArn: string;

  /**
   * The customer-managed KMS key used for the catalog's encryption at rest, if
   * one was configured.
   *
   * Undefined when encryption is disabled or an AWS-managed key is used. Grant
   * access to it via `KeyGrants`, e.g.
   * `if (catalog.encryptionKey) { KeyGrants.fromKey(catalog.encryptionKey).encrypt(grantee); }`.
   */
  readonly encryptionKey?: kms.IKeyRef;

  /**
   * The customer-managed KMS key used to encrypt connection passwords, if one
   * was configured.
   *
   * Undefined when password encryption uses an AWS-managed key or is not
   * configured. Grant access to it via `KeyGrants`, e.g.
   * `if (catalog.connectionPasswordKey) { KeyGrants.fromKey(catalog.connectionPasswordKey).encrypt(grantee); }`.
   */
  readonly connectionPasswordKey?: kms.IKeyRef;
}

/**
 * Base class for all `ICatalog` implementations. Materializes the single
 * `CfnDataCatalogEncryptionSettings` resource (targeting its own `catalogId`)
 * from the encryption options supplied at construction. Encryption is fixed at
 * construction, so a catalog either carries settings or it does not.
 */
export abstract class CatalogBase extends Resource implements ICatalog {
  public abstract readonly catalogId: string;
  public abstract readonly catalogArn: string;

  private _encryptionKey?: kms.IKeyRef;
  private _connectionPasswordKey?: kms.IKeyRef;

  public get encryptionKey(): kms.IKeyRef | undefined {
    return this._encryptionKey;
  }

  public get connectionPasswordKey(): kms.IKeyRef | undefined {
    return this._connectionPasswordKey;
  }

  public get catalogRef(): CatalogReference {
    return {
      resourceArn: this.catalogArn,
    };
  }

  /**
   * Emit the catalog's encryption settings from the options fixed at
   * construction. Subclasses call this once, after `catalogId`/`catalogArn` are
   * assigned. When neither block is configured, no resource is emitted, avoiding
   * an empty settings resource that would reset the catalog on deploy.
   */
  protected configureEncryption(options: CatalogEncryptionOptions): void {
    const atRest = options.encryptionAtRest;
    const password = options.connectionPasswordEncryption;

    if (!atRest && !password) {
      return;
    }

    this._encryptionKey = atRest?.kmsKey;
    this._connectionPasswordKey = password?.kmsKey;

    // Auto-grant the service role access to the customer-managed key it needs
    // to encrypt and decrypt catalog data. Nothing to grant for an AWS-managed
    // key (we don't own its key policy).
    if (atRest?.serviceRole && atRest.kmsKey) {
      KeyGrants.fromKey(atRest.kmsKey).encryptDecrypt(atRest.serviceRole);
    }

    // Two catalog instances that target the same catalog id would each emit a
    // settings resource and race to overwrite one another via
    // `PutDataCatalogEncryptionSettings`. Within one stack this is surfaced by
    // CloudFormation template validation (E3019: duplicate primary identifiers),
    // so we do not duplicate that check here.
    new CfnDataCatalogEncryptionSettings(this, 'EncryptionSettings', {
      catalogId: this.catalogId,
      dataCatalogEncryptionSettings: {
        encryptionAtRest: atRest
          ? {
            catalogEncryptionMode: atRest.mode,
            sseAwsKmsKeyId: atRest.kmsKey?.keyRef.keyArn,
            catalogEncryptionServiceRole: atRest.serviceRole?.roleArn,
          }
          : undefined,
        connectionPasswordEncryption: password
          ? {
            kmsKeyId: password.kmsKey?.keyRef.keyArn,
            returnConnectionPasswordEncrypted: password.returnConnectionPasswordEncrypted ?? true,
          }
          : undefined,
      },
    });
  }
}

/**
 * Construction properties for a `Catalog`.
 */
export interface CatalogProps extends CatalogEncryptionOptions {
  /**
   * The name of the catalog.
   */
  readonly catalogName: string;

  /**
   * A description of the catalog.
   *
   * @default - no description
   */
  readonly description?: string;
}

/**
 * The stack-scoped singleton id for the implicit account-wide catalog.
 */
const ACCOUNT_CATALOG_UID = '@aws-cdk.aws-glue-alpha.AccountCatalog';

/**
 * A Glue Data Catalog.
 *
 * Use `Catalog.forAccount(scope)` to obtain the implicit account-wide catalog,
 * `Catalog.encryptAccount(scope, options)` to configure its Data Catalog
 * encryption, or `new Catalog(...)` to create an `AWS::Glue::Catalog` resource.
 */
@propertyInjectable
export class Catalog extends CatalogBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-glue-alpha.Catalog';

  /**
   * Obtain the implicit, account-wide Data Catalog.
   *
   * The account catalog is not a CloudFormation resource; it always exists. This
   * returns a stack-scoped singleton, so repeated calls within the same stack
   * return the same instance.
   *
   * This returns the account catalog without managing its encryption. To
   * configure Data Catalog encryption for the account, use
   * `Catalog.encryptAccount(scope, options)` instead - it must be called before
   * the account catalog is first used in the stack.
   */
  public static forAccount(scope: Construct): ICatalog {
    const stack = Stack.of(scope);
    const existing = stack.node.tryFindChild(ACCOUNT_CATALOG_UID);
    return (existing as AccountCatalog) ?? new AccountCatalog(stack, ACCOUNT_CATALOG_UID, {});
  }

  /**
   * Configure Data Catalog encryption for the implicit, account-wide catalog and
   * return it.
   *
   * The account catalog's encryption is an account/region-wide setting, managed
   * through the singleton `PutDataCatalogEncryptionSettings` API. Because
   * encryption is fixed at construction, it must be configured before the
   * account catalog is first used in the stack: calling this after the account
   * catalog has already been materialized (for example by `Catalog.forAccount`,
   * or by a `Database` that uses the account catalog) throws.
   *
   * Configure it in exactly one stack. Configuring it from multiple stacks in the
   * same account and region makes those stacks overwrite one another at deploy
   * time, and the result is order-dependent. Unlike duplicate settings within a
   * single stack (which CloudFormation rejects), this cross-stack conflict is not
   * caught at synthesis time, because each stack synthesizes to its own template.
   */
  public static encryptAccount(scope: Construct, options: CatalogEncryptionOptions): ICatalog {
    const stack = Stack.of(scope);
    if (stack.node.tryFindChild(ACCOUNT_CATALOG_UID)) {
      throw new ValidationError(
        lit`AccountCatalogAlreadyInUse`,
        'the account catalog has already been used in this stack; call Catalog.encryptAccount() before Catalog.forAccount() or any Database that uses the account catalog',
        scope,
      );
    }
    return new AccountCatalog(stack, ACCOUNT_CATALOG_UID, options);
  }

  /**
   * Import an existing catalog by its ARN.
   *
   * The ARN must be a Glue catalog ARN, either the account-wide catalog
   * (`arn:aws:glue:<region>:<account>:catalog`, whose id is the account) or a
   * named catalog (`arn:aws:glue:<region>:<account>:catalog/<name>`, whose id is
   * the name).
   *
   * The imported catalog is a pure identity handle and does not manage the
   * catalog's encryption. To manage an existing catalog's Data Catalog
   * encryption, add a `CfnDataCatalogEncryptionSettings` resource targeting its
   * id.
   */
  public static fromCatalogArn(scope: Construct, id: string, catalogArn: string): ICatalog {
    const stack = Stack.of(scope);
    const arn = stack.splitArn(catalogArn, ArnFormat.SLASH_RESOURCE_NAME);

    // Only validate the shape of concrete ARNs; a tokenized ARN can't be
    // inspected at synth time, so we trust it.
    if (!Token.isUnresolved(catalogArn) && (arn.service !== 'glue' || arn.resource !== 'catalog')) {
      throw new ValidationError(
        lit`InvalidCatalogArn`,
        `expected a Glue catalog ARN (arn:<partition>:glue:<region>:<account>:catalog[/<name>]), got ${JSON.stringify(catalogArn)}`,
        scope,
      );
    }

    // The account-wide catalog's ARN has no resource name; its id is the
    // account from the ARN itself (falling back to the stack account for a
    // tokenized ARN with no parseable account).
    const catalogId = arn.resourceName ?? arn.account ?? stack.account;
    return new ImportedCatalog(scope, id, catalogId, catalogArn);
  }

  /**
   * Import an existing catalog by its id.
   *
   * The imported catalog is a pure identity handle and does not manage the
   * catalog's encryption. To manage an existing catalog's Data Catalog
   * encryption, add a `CfnDataCatalogEncryptionSettings` resource targeting its
   * id.
   */
  public static fromCatalogId(scope: Construct, id: string, catalogId: string): ICatalog {
    const stack = Stack.of(scope);
    const catalogArn = stack.formatArn({ service: 'glue', resource: 'catalog', resourceName: catalogId });
    return new ImportedCatalog(scope, id, catalogId, catalogArn);
  }

  public readonly catalogId: string;
  public readonly catalogArn: string;

  private readonly resource: CfnCatalog;

  constructor(scope: Construct, id: string, props: CatalogProps) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.resource = new CfnCatalog(this, 'Resource', {
      name: props.catalogName,
      description: props.description,
    });

    this.catalogId = this.resource.attrCatalogId;
    this.catalogArn = this.resource.attrResourceArn;

    this.configureEncryption(props);
  }
}

/**
 * The implicit, account-wide Data Catalog. Not a CloudFormation resource; only
 * the encryption settings it carries are synthesized.
 */
@propertyInjectable
class AccountCatalog extends CatalogBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-glue-alpha.AccountCatalog';
  public readonly catalogId: string;
  public readonly catalogArn: string;

  constructor(scope: Construct, id: string, encryption: CatalogEncryptionOptions) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, encryption);
    const stack = Stack.of(this);
    this.catalogId = stack.account;
    // The account catalog's id is implicitly the account id, so the ARN has no resource name.
    this.catalogArn = stack.formatArn({ service: 'glue', resource: 'catalog' });

    this.configureEncryption(encryption);
  }
}

/**
 * An imported catalog. A pure identity handle: it emits no resources and does
 * not manage the imported catalog's encryption.
 */
@propertyInjectable
class ImportedCatalog extends CatalogBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-glue-alpha.ImportedCatalog';
  public readonly catalogId: string;
  public readonly catalogArn: string;

  constructor(scope: Construct, id: string, catalogId: string, catalogArn: string) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, catalogId);
    this.catalogId = catalogId;
    this.catalogArn = catalogArn;
  }
}
