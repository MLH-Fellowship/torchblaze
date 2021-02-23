module.exports = {
  title: 'TorchBlaze',
  tagline: 'The only MLOps tool you\'ll need, from training to deployment.',
  url: 'https://github.com/MLH-Fellowship.github.io',
  baseUrl: '/torchblaze/',
  onBrokenLinks: 'throw',
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
          to: 'docs/',
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
              label: 'Style Guide',
              to: 'docs/',
            },
            {
              label: 'Second Doc',
              to: 'docs/doc2/',
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
