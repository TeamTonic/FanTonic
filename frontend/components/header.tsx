import Image from "next/image";
import { ModeToggle } from "./mode-toggle";
import { Button } from "./ui/button";
import Link from "next/link";

export const Header = () => {
    return (
        <header className="flex justify-between items-center w-full px-[5vw] border-b py-1.5">
            <Link href={"/"} className="flex items-center cursor-pointer gap-x-1.5 hover:opacity-85">
                <Image src={"/logo.png"} width={80} height={80} alt="Logo" />
                <div className="space-x-1.5 max-sm:hidden">
                    <span className="text-3xl font-black">Zang</span>
                    <span className="text-3xl font-black text-primary/90">beto</span>
                </div>
            </Link>
            <div className="flex gap-x-5">
                <ModeToggle />
                <Button className="text-base" asChild>
                    <Link href={"/dashboard"}>
                        Prompt in Fon now!
                    </Link>
                </Button>
            </div>
        </header>
    )
}