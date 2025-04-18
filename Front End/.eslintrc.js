// .eslintrc.js
module.exports = {
    root: true,
    env: {
      node: true,
      browser: true,
      es2021: true
    },
    extends: [
      'plugin:vue/vue3-recommended',
      'eslint:recommended'
    ],
    parserOptions: {
      parser: '@babel/eslint-parser',
      requireConfigFile: false,
      ecmaVersion: 2020,
      sourceType: 'module'
    },
    rules: {
      'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
      'no-debugger': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
      'vue/no-unused-components': 'warn',
      'vue/multi-word-component-names': 'off',
      'vue/no-v-html': 'off',
      'vue/require-default-prop': 'off',
      'vue/max-attributes-per-line': ['error', {
        singleline: {
          max: 4
        },
        multiline: {
          max: 1
        }
      }]
    }
  }