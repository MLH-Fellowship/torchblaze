module.exports = {
  title: 'TorchBlaze',
  tagline: 'The only MLOps tool you\'ll need, from training to deployment.',
  url: 'https://github.com/MLH-Fellowship.github.io',
  baseUrl: '/torchblaze/',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'MLH-Fellowship', // Usually your GitHub org/user name.
  projectName: 'torchblaze', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: 'TorchBlaze',
      logo: {
        alt: 'TorchBlaze Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          to: 'docs/installation',
          activeBasePath: 'docs',
          label: 'Docs',
          position: 'left',
        },
        {to: 'blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/MLH-Fellowship/torchblaze',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Installation',
              to: 'docs/installation/',
            },
            {
              label: 'Quick Setup',
              to: 'docs/setup/',
            },
            {
              label: 'Model Testing',
              to: 'docs/mltests/',
            },
            {
              label: 'API Testing',
              to: 'docs/apitest/',
            },
            {
              label: 'Quick Dockerizing',
              to: 'docs/docker/',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: 'blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/MLH-Fellowship/torchblaze',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} TorchBlaze, Inc. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/MLH-Fellowship/torchblaze/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/MLH-Fellowship/torchblaze/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
