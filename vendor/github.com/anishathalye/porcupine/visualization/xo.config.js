export default [
  {
    prettier: true,
    semicolon: false,
    space: true,
  },
  {
    files: ['*.js'],
    languageOptions: {
      globals: {
        document: 'readonly',
      },
    },
    rules: {
      '@stylistic/no-mixed-operators': 'off',
      'unicorn/prefer-module': 'off',
    },
  },
  {
    files: ['*.html'],
    rules: {
      '@html-eslint/attrs-newline': 'off',
      '@html-eslint/indent': ['error', 2],
      '@html-eslint/no-extra-spacing-tags': ['error', {enforceBeforeSelfClose: true}],
      '@html-eslint/require-closing-tags': ['error', {selfClosing: 'always'}],
      '@html-eslint/require-meta-description': 'off',
      '@html-eslint/require-open-graph-protocol': 'off',
      '@html-eslint/sort-attrs': 'off',
    },
  },
]
