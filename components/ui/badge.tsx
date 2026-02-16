import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/utils/cn"

const badgeVariants = cva(
    "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
    {
        variants: {
            variant: {
                default: "border-transparent bg-sky-500 text-white hover:bg-sky-600",
                secondary: "border-transparent bg-slate-100 text-slate-800 hover:bg-slate-200",
                destructive: "border-transparent bg-red-500 text-white hover:bg-red-600",
                outline: "text-slate-800 border-slate-300",
                success: "border-transparent bg-emerald-500 text-white hover:bg-emerald-600",
                warning: "border-transparent bg-amber-500 text-white hover:bg-amber-600",
            },
        },
        defaultVariants: { variant: "default" },
    }
)

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> { }

function Badge({ className, variant, ...props }: BadgeProps) {
    return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }