/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    optimizeCss: false
  },
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      '@tailwindcss/oxide': false
    }
    return config
  }
};

module.exports = nextConfig; 