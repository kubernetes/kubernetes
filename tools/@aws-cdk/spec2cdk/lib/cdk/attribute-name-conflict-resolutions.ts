import { attributePropertyName } from '../naming';

/**
 * Replacement names for attributes whose generated property name is already taken, keyed by resource
 * type and then by CloudFormation attribute name. Values are logical names without the `attr` prefix.
 *
 * Once a replacement name is published it is frozen, so entries are only ever added, never changed.
 *
 * `AWS::EKS::Cluster` needs one because `CertificateAuthority.Data` flattens onto the released
 * `attrCertificateAuthorityData`, which cannot move. The EKS API models it as a `Certificate` object,
 * so `CertificateAuthorityCertificateData` follows the service's own vocabulary.
 */
export const ATTRIBUTE_NAME_CONFLICT_RESOLUTIONS: Record<string, Record<string, string>> = {
  'AWS::EKS::Cluster': {
    'CertificateAuthority.Data': 'CertificateAuthorityCertificateData',
  },
};

/**
 * The property name for every attribute of a resource, keyed by CloudFormation attribute name.
 *
 * A recorded name always applies, whether or not the collision that caused it still exists: a
 * published name is frozen. An unrecorded collision is an error.
 */
export function attributePropertyNames(
  cloudFormationType: string,
  attributeNames: string[],
  resolutions: Record<string, Record<string, string>> = ATTRIBUTE_NAME_CONFLICT_RESOLUTIONS,
): Map<string, string> {
  const recorded = resolutions[cloudFormationType] ?? {};
  const propertyNames = new Map<string, string>();
  const attrNameByPropertyName = new Map<string, string>();

  for (const attrName of attributeNames) {
    const propertyName = attributePropertyName(recorded[attrName] ?? attrName);
    const owner = attrNameByPropertyName.get(propertyName);
    if (owner !== undefined) {
      const verb = recorded[owner] !== undefined || recorded[attrName] !== undefined ? 'Change' : 'Add';
      throw new Error(`Attribute name conflict on ${cloudFormationType} between '${owner}' and '${attrName}', which both become '${propertyName}'. ${verb} an entry in attribute-name-conflict-resolutions.ts for whichever of the two is NOT already published as '${propertyName}' in the latest released aws-cdk-lib - renaming the published one would repoint a released getter at a different Fn::GetAtt.`);
    }

    propertyNames.set(attrName, propertyName);
    attrNameByPropertyName.set(propertyName, attrName);
  }

  return propertyNames;
}
