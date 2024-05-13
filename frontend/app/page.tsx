"use client";
import { AnimatedTooltip } from "@/components/ui/animated-tooltip";
import { HeroHighlight, Highlight } from "@/components/ui/hero-highlight";
import { people } from "@/lib/constants";
import { motion } from "framer-motion";
import Link from "next/link";

export default function Home() {
  return (
    <main className="pt-36 max-sm:pt-10">
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
          Your Fan Language Chat Companion <br />
          <Link href={"/chatbot"}>
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
          <h2 className="text-5xl font-bold md:text-6xl pb-9">Meet our team!</h2>
          <div className="ml-[-1rem] flex flex-row justify-center">
            <AnimatedTooltip items={people} />
          </div>
        </motion.footer>
      </HeroHighlight>
    </main>
  );
}
