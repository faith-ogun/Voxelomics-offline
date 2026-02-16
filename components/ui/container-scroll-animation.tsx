import React, { useEffect, useRef, useState } from 'react';
import { motion, MotionValue, useScroll, useTransform } from 'framer-motion';

interface ContainerScrollProps {
  titleComponent: React.ReactNode;
  children: React.ReactNode;
}

export const ContainerScroll: React.FC<ContainerScrollProps> = ({ titleComponent, children }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef
  });
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const rotate = useTransform(scrollYProgress, [0, 1], [18, 0]);
  const scale = useTransform(scrollYProgress, [0, 1], isMobile ? [0.86, 1] : [1.06, 1]);
  const translate = useTransform(scrollYProgress, [0, 1], [0, -110]);

  return (
    <div
      ref={containerRef}
      className="h-[44rem] md:h-[58rem] flex items-center justify-center relative px-2 md:px-10"
    >
      <div
        className="py-10 md:py-20 w-full relative"
        style={{ perspective: '1000px' }}
      >
        <Header translate={translate} titleComponent={titleComponent} />
        <Card rotate={rotate} scale={scale}>
          {children}
        </Card>
      </div>
    </div>
  );
};

const Header = ({
  translate,
  titleComponent
}: {
  translate: MotionValue<number>;
  titleComponent: React.ReactNode;
}) => {
  return (
    <motion.div
      style={{ translateY: translate }}
      className="max-w-5xl mx-auto text-center px-4"
    >
      {titleComponent}
    </motion.div>
  );
};

const Card = ({
  rotate,
  scale,
  children
}: {
  rotate: MotionValue<number>;
  scale: MotionValue<number>;
  children: React.ReactNode;
}) => {
  return (
    <motion.div
      style={{
        rotateX: rotate,
        scale,
        boxShadow:
          '0 0 #0000004d, 0 9px 20px #0000004a, 0 37px 37px #00000042, 0 84px 50px #00000026, 0 149px 60px #0000000a, 0 233px 65px #00000003'
      }}
      className="max-w-5xl -mt-10 mx-auto h-[24rem] md:h-[36rem] w-full border-4 border-[#1a2a82] p-2 md:p-4 bg-[#10207c] rounded-[26px] shadow-2xl"
    >
      <div className="h-full w-full overflow-hidden rounded-2xl bg-slate-950 md:rounded-2xl">
        {children}
      </div>
    </motion.div>
  );
};
