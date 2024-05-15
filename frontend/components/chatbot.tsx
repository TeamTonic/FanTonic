"use client";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "./ui/button";
import { Form, FormControl, FormDescription, FormField, FormItem, FormMessage } from "./ui/form";
import { Input } from "./ui/input";
import { useState } from "react";

const formSchema = z.object({
    prompt: z.string().min(10, {
        message: "Minimum length of your question should be 10 characters."
    })
})

export const Chatbot = () => {
    const [responses, setResponses] = useState([]);

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            prompt: "",
        },
    })

    function onSubmit(values: z.infer<typeof formSchema>) {
        console.log(values);
    }

    return (
        <section className="flex flex-col justify-between py-10 min-h-[680px]">
            <div className="flex flex-col gap-y-5">
                <div className="px-5 py-3 rounded-lg bg-secondary">messages here.</div>
            </div>

            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                    <FormField
                        control={form.control}
                        name="prompt"
                        render={({ field }) => (
                            <FormItem>
                                <FormControl>
                                    <Input placeholder="What language do people speek in south Benin?" {...field} />
                                </FormControl>
                                <FormDescription>
                                    Ask whatever you want to learn about Benin.
                                </FormDescription>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <Button type="submit">Submit</Button>
                </form>
            </Form>
        </section>
    )
}