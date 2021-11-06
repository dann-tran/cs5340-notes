import { Metadata } from 'lib/models';
import Head from 'next/head';
import Link from 'next/link';
import React from 'react';
import styles from './Layout.module.css';

const Layout: React.FC<{
  title: string;
  children: React.ReactNode;
  metadataList: Metadata[];
}> = ({ title, children, metadataList }) => {
  return (
    <div className='Layout'>
      <Head>
        <title>{title}</title>
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <meta charSet='utf-8' />
      </Head>

      <main className={styles['layout-container']}>
        <nav className={styles['navbar']}>
          <ul>
            <li>
              <Link href='/'>
                <a>Home</a>
              </Link>
            </li>
            <li>
              <Link href='https://github.com/picasdan9/cs5340-notes'>
                <a target='_blank'>Github</a>
              </Link>
            </li>
          </ul>
          <div className={styles['navbar-contents']}>
            <div className={styles['navbar-contents-header']}>Contents</div>
            <ol>
              {metadataList.map((metadata) => (
                <li key={metadata.slug}>
                  <Link href={metadata.slug}>
                    <a>{metadata.title}</a>
                  </Link>
                </li>
              ))}
            </ol>
          </div>
        </nav>
        <div className={styles['page-container']}>{children}</div>
      </main>
    </div>
  );
};

export default Layout;
