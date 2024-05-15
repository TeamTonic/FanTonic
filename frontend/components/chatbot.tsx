"use client";
import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { ArrowUp, Mic } from "lucide-react";
import { Button } from "./ui/button";
import { Form, FormControl, FormField, FormItem, FormMessage } from "./ui/form";
import { Input } from "./ui/input";

const formSchema = z.object({
    prompt: z.string().min(10, {
        message: "Minimum length of your question should be 10 characters."
    })
})

export const Chatbot = () => {
    const [responses, setResponses] = useState<any>([]);

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            prompt: "",
        },
    })

    async function onSubmit(values: z.infer<typeof formSchema>) {
        try {
            const response = await fetch("http://127.0.0.1:5000/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ "query_string": values.prompt }),
            });

            const data = await response.json();

            setResponses((prevResponses: any) => [
                ...prevResponses,
                { question: values.prompt },
                { answer: data.text },
            ]);

            form.reset();
        } catch (err: any) {
            throw new Error(err.message);
        }
    }

    return (
        <section className="flex flex-col justify-between py-10 min-h-[720px]">
            <div className="flex flex-col gap-y-3">
                {responses.map((response: any, index: any) => (
                    <div key={index}>
                        {response.question &&
                            <div className="px-5 py-4 mt-3 ml-auto rounded-full bg-secondary">{response.question}</div>
                        }
                        {response.answer &&
                            <div className="px-5 py-4 rounded-full">{response.answer}</div>
                        }
                    </div>
                ))}
            </div>

            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="relative">
                    <FormField
                        control={form.control}
                        name="prompt"
                        render={({ field }) => (
                            <FormItem className="flex flex-col">
                                <FormControl>
                                    <Input
                                        placeholder="What language do people speek in south Benin?"
                                        className="py-8 pl-6 pr-10 text-lg rounded-full bg-secondary"
                                        {...field}
                                    />
                                </FormControl>
                                <FormMessage className="pl-4 text-base" />
                            </FormItem>
                        )}
                    />
                    <div className="flex justify-end gap-3 mt-4">
                        <Button type="submit" disabled={!form.getValues("prompt")} className="p-3 py-6 rounded-full">
                            <ArrowUp />
                        </Button>
                        <Button type="button" className="p-3 py-6 rounded-full">
                            <Mic />
                        </Button>
                    </div>
                </form>
            </Form>
        </section>
    )
}