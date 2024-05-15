"use client";
import { AnimatedTooltip } from "@/components/ui/animated-tooltip";
import { HeroHighlight, Highlight } from "@/components/ui/hero-highlight";
import { globeConfig, people, sampleArcs } from "@/lib/constants";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import Link from "next/link";

export default function Home() {
  const World = dynamic(() => import("@/components/ui/globe").then((m) => m.World), {
    ssr: false,
  });

  return (
    <main className="pt-44 max-sm:pt-10">
      <HeroHighlight>
        <motion.h1
          initial={{
            opacity: 0,
            y: 20,
          }}
          animate={{
            opacity: 1,
            y: [20, -5, 0],
          }}
          transition={{
            duration: 0.5,
            ease: [0.4, 0.0, 0.2, 1],
          }}
          className="max-w-4xl px-4 mx-auto text-5xl font-bold leading-relaxed text-center md:text-6xl lg:text-7xl lg:leading-loose text-neutral-700 dark:text-white"
        >
          Your Fon Language Chat Companion <br />
          <Link href={"/dashboard"}>
            <Highlight className="p-5 mt-20 text-4xl">
              Try it now!
            </Highlight>
          </Link>
        </motion.h1>
      </HeroHighlight>
      <HeroHighlight>
        <motion.footer
          initial={{
            opacity: 0,
            y: 20,
          }}
          animate={{
            opacity: 1,
            y: [20, -5, 0],
          }}
          transition={{
            duration: 0.8,
            ease: [0.5, 0.0, 0.4, 1],
          }}
        >
          <h2 className="text-4xl font-bold md:text-5xl pb-9">Meet our team!</h2>
          <div className="ml-[-1rem] flex flex-row justify-center">
            <AnimatedTooltip items={people} />
          </div>
        </motion.footer>
      </HeroHighlight>
      <div className="mx-auto w-full relative overflow-hidden h-[42rem] max-sm:h-[25rem] px-4">
        <motion.div
          initial={{
            opacity: 0,
            y: 20,
          }}
          animate={{
            opacity: 1,
            y: 0,
          }}
          transition={{
            duration: 1,
          }}
          className="div"
        >
          <h2 className="text-xl font-bold text-center text-black md:text-4xl dark:text-white">
            Use Zangbeto wherever
          </h2>
          <p className="max-w-md mx-auto mt-2 text-base font-normal text-center md:text-lg text-neutral-700 dark:text-neutral-200">
            Have fun with our app, use it for studying purposes or even agroculture.
          </p>
        </motion.div>
        <div className="absolute inset-x-0 bottom-0 z-40 w-full h-40 pointer-events-none select-none bg-gradient-to-b from-transparent dark:to-black to-white" />
        <div className="absolute z-10 w-full h-full mb-10">
          <World data={sampleArcs} globeConfig={globeConfig} />;
        </div>
      </div>
    </main>
  );
}
