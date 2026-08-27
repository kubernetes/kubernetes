import { attributePropertyNames } from '../lib/cdk/attribute-name-conflict-resolutions';

const RESOURCE_TYPE = 'AWS::Some::Resource';
const RESOLUTIONS = {
  [RESOURCE_TYPE]: {
    'Foo.Bar': 'FooBarValue',
  },
};

const names = (attributeNames: string[], resolutions: Record<string, Record<string, string>> = RESOLUTIONS) =>
  Object.fromEntries(attributePropertyNames(RESOURCE_TYPE, attributeNames, resolutions));

test.each([
  ['flat attribute first', ['FooBar', 'Foo.Bar']],
  ['nested attribute first', ['Foo.Bar', 'FooBar']],
])('the flat attribute keeps the preferred name and the nested one takes its recorded name (%s)', (_name, attributeNames) => {
  expect(names(attributeNames)).toEqual({
    'FooBar': 'attrFooBar',
    'Foo.Bar': 'attrFooBarValue',
  });
});

test('a recorded name still applies once the attribute it collided with is gone', () => {
  // The recorded name is published, so it must not revert to the preferred name that is now free
  expect(names(['Foo.Bar'])).toEqual({ 'Foo.Bar': 'attrFooBarValue' });
});

test('attributes that do not conflict all keep their preferred names', () => {
  expect(names(['Arn', 'Foo.Baz'])).toEqual({
    'Arn': 'attrArn',
    'Foo.Baz': 'attrFooBaz',
  });
});

const RULE = "for whichever of the two is NOT already published as '%s' in the latest released aws-cdk-lib "
  + '- renaming the published one would repoint a released getter at a different Fn::GetAtt.';

test('an attribute name conflict with no recorded resolution throws, naming neither as the one to rename', () => {
  expect(() => names(['FooBar', 'Foo.Bar'], {})).toThrow(
    "Attribute name conflict on AWS::Some::Resource between 'FooBar' and 'Foo.Bar', which both become 'attrFooBar'. "
    + 'Add an entry in attribute-name-conflict-resolutions.ts ' + RULE.replace('%s', 'attrFooBar'),
  );
});

test('the conflict message does not single out the nested attribute, which may be the published one', () => {
  // A published nested attribute must not be named as the one to rename
  let message = '';
  try {
    names(['FooBarBaz', 'Foo.Bar.Baz'], {});
  } catch (e) {
    message = (e as Error).message;
  }

  expect(message).toContain('for whichever of the two is NOT already published');
  expect(message).not.toContain("rename 'Foo.Bar.Baz'");
});

test('a conflict on an attribute that already has an entry says to change it, not add one', () => {
  // Foo.Bar is recorded as FooBarValue, which FooBarValue's own preferred name already owns
  expect(() => names(['FooBar', 'FooBarValue', 'Foo.Bar'])).toThrow(
    "Attribute name conflict on AWS::Some::Resource between 'FooBarValue' and 'Foo.Bar', which both become 'attrFooBarValue'. "
    + 'Change an entry in attribute-name-conflict-resolutions.ts ' + RULE.replace('%s', 'attrFooBarValue'),
  );
});
