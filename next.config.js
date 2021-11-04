/** @type {import('next').NextConfig} */
const isProd = process.env.NODE_ENV === 'production';

module.exports = {
  reactStrictMode: true,
  basePath: isProd ? '/cs5340-notes' : '',
  assetPrefix: isProd ? '/cs5340-notes' : '',
};
